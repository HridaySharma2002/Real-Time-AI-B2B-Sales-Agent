"""
Rag.py - Real-Time AI B2B Sales Agent RAG Module
Integrated with local Ollama running `qwen:0.5b` model and external `Rag_Knowledge_base.txt`.

Features:
- Connects to local Ollama API (http://localhost:11434) using `qwen:0.5b`.
- Loads all sales domain knowledge dynamically from external `Rag_Knowledge_base.txt`.
- Real-time streaming response generation optimized for low-latency sales conversations.
- Vector retrieval system supporting embedding-based similarity search (Ollama embeddings,
  SentenceTransformers, or resilient built-in vector similarity).
- Extensible API for indexing external files, markdown, PDFs, or raw text.
- Interactive CLI for live testing.
"""

import os
import sys
import json
import time
import math
import ast
import re
import logging
import asyncio
import urllib.request
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SalesRAG")

# =====================================================================
# Configuration & Defaults
# =====================================================================

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen:0.5b")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_KB_FILE = os.getenv(
    "KNOWLEDGE_BASE_FILE",
    os.path.join(os.path.dirname(__file__), "Rag_Knowledge_base.txt")
)


@dataclass
class DocumentChunk:
    """Represents a chunk of knowledge in the vector database."""
    id: str
    content: str
    category: str  # e.g., 'pricing', 'objection_handling', 'product_feature', 'battlecard'
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


# =====================================================================
# FAISS Vector Database & Dense Embedding Engine
# =====================================================================

class FAISSVectorRetriever:
    """
    High-speed FAISS vector database and dense embedding retrieval engine.
    Utilizes FAISS IndexFlatIP (Inner Product / Normalized Cosine Similarity)
    for sub-millisecond retrieval of B2B sales knowledge chunks.
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        use_sentence_transformers: bool = True
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_model = embed_model
        self.chunks: List[DocumentChunk] = []
        self.dimension: Optional[int] = None
        
        # FAISS index instance
        self.faiss_index = None
        self._faiss_available = False
        self._st_model = None
        
        self._init_embedding_model(use_sentence_transformers)
        self._init_faiss()

    def _init_embedding_model(self, use_sentence_transformers: bool):
        """Initializes fast local SentenceTransformer or falls back to Ollama / Hash vectors."""
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded local SentenceTransformer ('all-MiniLM-L6-v2') for high-speed embeddings.")
                return
            except Exception as e:
                logger.debug(f"SentenceTransformer not loaded ({e}). Using Ollama/fallback embeddings.")
                self._st_model = None

    def _init_faiss(self):
        """Attempts to load FAISS library."""
        try:
            import faiss
            self._faiss = faiss
            self._faiss_available = True
            logger.info("FAISS vector database engine initialized successfully.")
        except ImportError:
            self._faiss = None
            self._faiss_available = False
            logger.info("FAISS library not found. Using high-speed vectorized NumPy cosine engine.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Computes dense float32 vector embedding for text."""
        # 1. Local SentenceTransformer (fastest, CPU/GPU local)
        if self._st_model is not None:
            emb = self._st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)

        # 2. Ollama /api/embeddings endpoint
        try:
            url = f"{self.ollama_url}/api/embeddings"
            payload = json.dumps({"model": self.embed_model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                emb = result.get("embedding")
                if emb:
                    arr = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    return arr
        except Exception:
            pass

        # 3. Normalized hash vector fallback
        vec_dim = 384
        vec = np.zeros(vec_dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            h = hash(word) % vec_dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def build_faiss_index(self):
        """Builds or rebuilds the FAISS IndexFlatIP from all indexed chunks."""
        if not self.chunks:
            return

        # Prepare embedding matrix
        embeddings = [c.embedding for c in self.chunks]
        matrix = np.vstack(embeddings).astype(np.float32)
        self.dimension = matrix.shape[1]

        if self._faiss_available:
            # Normalize vectors for exact cosine similarity with Inner Product
            self._faiss.normalize_L2(matrix)
            # IndexFlatIP calculates inner product on normalized vectors = cosine similarity
            self.faiss_index = self._faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(matrix)
            logger.info(f"Built FAISS IndexFlatIP with {self.faiss_index.ntotal} vectors (dim={self.dimension}).")
        else:
            self.faiss_index = matrix

    def add_chunk(self, chunk: DocumentChunk):
        """Adds a single chunk and computes its embedding."""
        if chunk.embedding is None:
            emb = self.get_embedding(f"{chunk.category} {chunk.content}")
            chunk.embedding = emb
        self.chunks.append(chunk)

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Batch adds documents and rebuilds FAISS index for instant querying."""
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            category = doc.get("category", "general")
            metadata = doc.get("metadata", {})
            if "title" in doc:
                metadata["title"] = doc["title"]
            chunk = DocumentChunk(
                id=f"doc_{len(self.chunks) + 1}",
                content=content,
                category=category,
                metadata=metadata
            )
            emb = self.get_embedding(f"{category} {content}")
            chunk.embedding = emb
            self.chunks.append(chunk)

        # Rebuild FAISS index with all embeddings
        self.build_faiss_index()
        logger.info(f"FAISS indexed {len(docs)} documents. Total active vectors: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 3, threshold: float = 0.05) -> List[DocumentChunk]:
        """
        Executes sub-millisecond FAISS similarity search returning top-k matching chunks.
        """
        if not self.chunks or self.faiss_index is None:
            return []

        query_vec = self.get_embedding(query).reshape(1, -1).astype(np.float32)
        scored_results: List[Tuple[float, DocumentChunk]] = []

        # 1. FAISS Search
        if self._faiss_available and hasattr(self.faiss_index, "search"):
            self._faiss.normalize_L2(query_vec)
            k = min(top_k * 2, len(self.chunks))
            distances, indices = self.faiss_index.search(query_vec, k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    score = float(dist)
                    
                    # Sales keyword boost
                    query_words = set(re.findall(r'\w+', query.lower()))
                    chunk_words = set(re.findall(r'\w+', chunk.content.lower()))
                    overlap = len(query_words.intersection(chunk_words))
                    boosted_score = score + (0.05 * min(overlap, 5))

                    if boosted_score >= threshold:
                        scored_results.append((boosted_score, chunk))
        else:
            # High-speed vectorized NumPy fallback
            matrix = self.faiss_index  # shape (N, dim)
            scores = np.dot(matrix, query_vec.T).flatten()
            for idx, score in enumerate(scores):
                chunk = self.chunks[idx]
                query_words = set(re.findall(r'\w+', query.lower()))
                chunk_words = set(re.findall(r'\w+', chunk.content.lower()))
                overlap = len(query_words.intersection(chunk_words))
                boosted_score = float(score) + (0.05 * min(overlap, 5))
                if boosted_score >= threshold:
                    scored_results.append((boosted_score, chunk))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_results[:top_k]]


# Backwards compatibility alias
VectorRetriever = FAISSVectorRetriever


# =====================================================================
# Local Ollama Client & Sales RAG Agent
# =====================================================================

class B2BSalesRAG:
    """
    Main RAG agent for Real-Time B2B Sales conversations using local Ollama model `qwen:0.5b`
    and external knowledge base from `Rag_Knowledge_base.txt`.
    """

    SYSTEM_PROMPT = (
        "You are an expert, professional, and friendly B2B Sales Executive representing ApexSales AI. "
        "Your mission is to engage prospective clients in a natural, consultative, and persuasive sales conversation.\n\n"
        "GUIDELINES:\n"
        "1. Consultative & Helpful: Actively listen, address business pains, and build trust.\n"
        "2. Grounded Answers: Use the provided Context from our sales knowledge base to answer questions accurately about "
        "features, pricing, security, and integrations. Do NOT make up false claims.\n"
        "3. Objection Handling: When faced with hesitation or objections, empathize, reframe with ROI and proof points, "
        "and keep the conversation moving forward.\n"
        "4. Concise & Punchy: In a real-time conversation or voice call, avoid overwhelming walls of text. Keep responses "
        "concise (2 to 4 sentences), impactful, and end with an engaging open question or clear call-to-action (e.g., booking a 15-min demo)."
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        knowledge_base_file: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.retriever = VectorRetriever(ollama_url=self.ollama_url, embed_model=embed_model)
        self.conversation_history: List[Dict[str, str]] = []

        # Load knowledge base exclusively from external file
        self.kb_file = knowledge_base_file or DEFAULT_KB_FILE
        self.load_knowledge_base_from_file(self.kb_file)

    def load_knowledge_base_from_file(self, filepath: str):
        """Loads and indexes documents from the given knowledge base file."""
        if not os.path.exists(filepath):
            logger.warning(f"Knowledge base file not found at '{filepath}'.")
            return

        docs = self.parse_knowledge_base_file(filepath)
        if docs:
            self.retriever.add_documents(docs)
            logger.info(f"Loaded {len(docs)} knowledge chunks from {filepath}")
        else:
            logger.warning(f"No valid knowledge chunks parsed from {filepath}")

    @staticmethod
    def parse_knowledge_base_file(filepath: str) -> List[Dict[str, Any]]:
        """
        Parses knowledge base files supporting:
        1. Structured section tags ([SECTION: CATEGORY], Title:, Category:, Content:)
        2. Python list of dicts (if python code structure is present)
        3. JSON arrays
        4. Standard Markdown / Paragraphs
        """
        docs = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                return []

            # 1. Try Python list/dict parsing (e.g., DEFAULT_SALES_KNOWLEDGE = [...])
            if "DEFAULT_SALES_KNOWLEDGE" in content or content.startswith("["):
                try:
                    py_code = content
                    if "DEFAULT_SALES_KNOWLEDGE" in content:
                        py_code = content.split("DEFAULT_SALES_KNOWLEDGE", 1)[1]
                        py_code = py_code.split("=", 1)[1].strip()
                    parsed_py = ast.literal_eval(py_code)
                    if isinstance(parsed_py, list):
                        for item in parsed_py:
                            if isinstance(item, dict) and "content" in item:
                                docs.append({
                                    "category": item.get("category", "general"),
                                    "title": item.get("title", ""),
                                    "content": item.get("content", ""),
                                    "metadata": item.get("metadata", {"source": os.path.basename(filepath)})
                                })
                        if docs:
                            return docs
                except Exception:
                    pass  # Fall through to section parser

            # 2. Try [SECTION: ...] tagged format
            if "[SECTION:" in content:
                raw_sections = content.split("[SECTION:")
                for section in raw_sections:
                    if not section.strip() or section.startswith("="):
                        continue
                    lines = section.strip().splitlines()
                    header = lines[0].split("]")[0].strip().lower()
                    category = header if header else "general"
                    
                    title = ""
                    doc_content_lines = []
                    is_reading_content = False

                    for line in lines[1:]:
                        line_str = line.strip()
                        if line_str.startswith("Title:"):
                            title = line_str.replace("Title:", "").strip()
                        elif line_str.startswith("Category:"):
                            category = line_str.replace("Category:", "").strip().lower()
                        elif line_str.startswith("Content:"):
                            is_reading_content = True
                        elif is_reading_content:
                            doc_content_lines.append(line)

                    full_text = "\n".join(doc_content_lines).strip()
                    if full_text:
                        docs.append({
                            "category": category,
                            "title": title or category,
                            "content": full_text,
                            "metadata": {"source": os.path.basename(filepath)}
                        })
                if docs:
                    return docs

            # 3. Fallback: Parse paragraphs separated by blank lines
            paragraphs = content.split("\n\n")
            for i, p in enumerate(paragraphs):
                p_text = p.strip()
                if p_text and not p_text.startswith("="):
                    docs.append({
                        "category": "general",
                        "title": f"Section {i+1}",
                        "content": p_text,
                        "metadata": {"source": os.path.basename(filepath)}
                    })

        except Exception as e:
            logger.error(f"Error reading knowledge base from {filepath}: {e}")

        return docs

    def check_ollama_connection(self) -> Dict[str, Any]:
        """Verifies connection to local Ollama server and checks if requested model exists."""
        url = f"{self.ollama_url}/api/tags"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                model_found = any(self.model_name in m for m in models)
                return {
                    "connected": True,
                    "available_models": models,
                    "target_model": self.model_name,
                    "model_available": model_found,
                    "status": "Ready" if model_found else f"Model '{self.model_name}' not downloaded yet. Run `ollama pull {self.model_name}`."
                }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "target_model": self.model_name,
                "model_available": False,
                "status": f"Ollama not reachable at {self.ollama_url}. Please ensure Ollama is running (`ollama serve`)."
            }

    def load_custom_documents(self, docs: List[Dict[str, Any]]):
        """Ingests additional custom sales collateral, battlecards, or FAQs."""
        self.retriever.add_documents(docs)

    def build_prompt(self, user_query: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """Constructs grounded context prompt for Qwen 0.5B."""
        context_texts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            category = chunk.category.replace("_", " ").title()
            context_texts.append(f"[{category} {i}]: {chunk.content}")

        context_str = "\n".join(context_texts) if context_texts else "No specific knowledge base context found."

        # Build prompt with history
        prompt = f"<|im_start|>system\n{self.system_prompt}\n\n[RELEVANT SALES KNOWLEDGE BASE CONTEXT]:\n{context_str}<|im_end|>\n"
        
        # Include recent turns of conversation
        for msg in self.conversation_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def generate(self, user_query: str, top_k: int = 3, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Synchronous full response generation with RAG retrieval.
        Returns generated text, retrieved context, and latency metrics.
        """
        start_time = time.time()
        
        # 1. Retrieve knowledge
        retrieved_chunks = self.retriever.search(user_query, top_k=top_k)
        retrieval_time = time.time() - start_time

        # 2. Build prompt
        prompt = self.build_prompt(user_query, retrieved_chunks)

        # 3. Call local Ollama
        gen_start = time.time()
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["<|im_end|>", "<|im_start|>"]
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "").strip()
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            response_text = (
                f"[Error connecting to Ollama at {self.ollama_url}]. "
                f"Please ensure Ollama is running and model '{self.model_name}' is pulled (`ollama run {self.model_name}`)."
            )

        total_time = time.time() - start_time
        gen_time = time.time() - gen_start

        # Record conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return {
            "query": user_query,
            "response": response_text,
            "retrieved_context": [
                {"category": c.category, "content": c.content, "metadata": c.metadata}
                for c in retrieved_chunks
            ],
            "metrics": {
                "retrieval_ms": round(retrieval_time * 1000, 2),
                "generation_ms": round(gen_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2)
            }
        }

    def generate_stream(self, user_query: str, top_k: int = 3, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Real-time streaming token generator for low-latency speech/voice synthesis pipelines.
        Yields tokens as they arrive from Ollama.
        """
        # 1. Retrieve knowledge
        retrieved_chunks = self.retriever.search(user_query, top_k=top_k)
        
        # 2. Build prompt
        prompt = self.build_prompt(user_query, retrieved_chunks)

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["<|im_end|>", "<|im_start|>"]
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        full_response = []
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp:
                    if line:
                        chunk_json = json.loads(line.decode("utf-8"))
                        token = chunk_json.get("response", "")
                        if token:
                            full_response.append(token)
                            yield token
                        if chunk_json.get("done", False):
                            break
        except Exception as e:
            err_msg = f"\n[Streaming Error: {e}]"
            full_response.append(err_msg)
            yield err_msg

        # Update history
        complete_text = "".join(full_response).strip()
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": complete_text})

    def reset_conversation(self):
        """Clears the current conversation memory."""
        self.conversation_history = []
        logger.info("Conversation history reset.")


# =====================================================================
# CLI Demo / Test Runner
# =====================================================================

def interactive_cli():
    """Interactive command-line interface to test the B2B Sales RAG agent."""
    print("=" * 70)
    print("  Real-Time AI B2B Sales Agent - Local RAG (qwen:0.5b via Ollama)")
    print("=" * 70)

    agent = B2BSalesRAG()
    
    # Check Ollama status
    print("\n[1/2] Checking local Ollama connection...")
    status = agent.check_ollama_connection()
    if status["connected"]:
        print(f"  ✓ Connected to Ollama at {agent.ollama_url}")
        if status["model_available"]:
            print(f"  ✓ Model '{agent.model_name}' is available and ready!")
        else:
            print(f"  ! Model '{agent.model_name}' was not detected in local models list: {status['available_models']}")
            print(f"    To pull the model, run: ollama pull {agent.model_name}")
    else:
        print(f"  ✗ Ollama is not reachable ({status.get('error')}).")
        print("    Please ensure Ollama is installed and running (`ollama serve`).")

    print(f"\n[2/2] Knowledge Base Loaded from '{os.path.basename(agent.kb_file)}': {len(agent.retriever.chunks)} chunks.")
    print("\nCommands:")
    print("  'exit' or 'quit' : Exit interactive chat")
    print("  'clear'          : Clear conversation memory")
    print("  'stream on/off'  : Toggle streaming mode (default: ON)")
    print("=" * 70 + "\n")

    streaming = True

    while True:
        try:
            user_input = input("\nProspect > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting B2B Sales Agent. Goodbye!")
                break
            if user_input.lower() == "clear":
                agent.reset_conversation()
                print("Memory cleared.")
                continue
            if user_input.lower() == "stream on":
                streaming = True
                print("Streaming enabled.")
                continue
            if user_input.lower() == "stream off":
                streaming = False
                print("Streaming disabled.")
                continue

            print("\nAI Sales Rep > ", end="", flush=True)

            if streaming:
                for token in agent.generate_stream(user_input):
                    print(token, end="", flush=True)
                print()
            else:
                result = agent.generate(user_input)
                print(result["response"])
                print(f"\n[Latency: Total {result['metrics']['total_ms']}ms | RAG {result['metrics']['retrieval_ms']}ms | Gen {result['metrics']['generation_ms']}ms]")
                if result["retrieved_context"]:
                    print(f"[Context retrieved: {len(result['retrieved_context'])} chunk(s)]")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    interactive_cli()
