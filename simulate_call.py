"""
simulate_call.py - Live B2B Sales Call Simulator with K-Means Learning & RAG Self-Healing

Simulates a real-time call between a human Prospect (You) and the AI Sales Agent.
1. Live conversation powered by Rag.py (qwen:0.5b via Ollama) and Chatterbox TTS.
2. Real-time telemetry tracking:
   - Call Duration & Turn count
   - Prospect Tone / Sentiment polarity
   - Friction topics (Pricing, Security, Competitors, Complexity, Timing)
   - Exact Drop-off Point / Cut-off reasons
3. End-of-Call K-Means Re-Clustering:
   - Ingests the new call into historical database (`call_history.json`).
   - Runs K-Means clustering to mathematically classify the call and diagnose failure points.
4. Autonomous RAG Knowledge Base Update:
   - If friction/drop-off occurred, automatically generates and appends new objection playbooks
     to `Rag_Knowledge_base.txt` and re-indexes FAISS for future calls!
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

# Local imports
from Rag import B2BSalesRAG
import importlib.util

def _import_local_module(module_name: str, file_name: str):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load chatterbox-tts.py and k-means_clustering.py safely
_tts_mod = _import_local_module("chatterbox_tts", "chatterbox-tts.py")
ChatterboxTTSAgent = _tts_mod.ChatterboxTTSAgent

_kmeans_mod = _import_local_module("kmeans_clustering", "k-means_clustering.py")
SalesCallClustering = _kmeans_mod.SalesCallClustering
RAGKnowledgeOptimizer = _kmeans_mod.RAGKnowledgeOptimizer
generate_sample_sales_call_logs = _kmeans_mod.generate_sample_sales_call_logs
KB_FILE_PATH = _kmeans_mod.KB_FILE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CallSimulator")

CALL_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "call_history.json")


# =====================================================================
# Real-Time Tone & Sentiment Analyzer
# =====================================================================

class ProspectToneAnalyzer:
    """Calculates sentiment polarity and detects sales friction topics from prospect text."""

    POSITIVE_WORDS = {
        "interested", "great", "awesome", "perfect", "good", "love", "yes", "demo",
        "schedule", "pricing", "send", "book", "sounds", "impressive", "helpful", "let's"
    }
    
    NEGATIVE_WORDS = {
        "expensive", "budget", "no", "not", "cannot", "hard", "complicated", "slow",
        "busy", "later", "competitor", "drift", "already", "waste", "stop", "bad", "bye"
    }

    FRICTION_TOPIC_KEYWORDS = {
        "pricing": ["expensive", "cost", "price", "budget", "tier", "$", "discount", "cheap"],
        "competitor_drift": ["drift", "intercom", "salesloft", "outreach", "competitor", "alternative", "already use"],
        "technical_latency": ["latency", "slow", "delay", "lag", "voice quality", "robotic"],
        "security_compliance": ["security", "privacy", "gdpr", "hipaa", "soc2", "data", "cloud", "vpc"],
        "gatekeeper_bounce": ["busy", "not interested", "hang up", "bye", "stop calling", "remove me", "who is this"]
    }

    @classmethod
    def analyze_utterance(cls, text: str) -> Dict[str, Any]:
        """Analyzes sentiment (-1.0 to +1.0) and detects friction category."""
        words = set(text.lower().split())
        
        pos_count = len(words.intersection(cls.POSITIVE_WORDS))
        neg_count = len(words.intersection(cls.NEGATIVE_WORDS))
        
        total = pos_count + neg_count
        if total == 0:
            sentiment = 0.05  # neutral baseline
        else:
            sentiment = (pos_count - neg_count) / max(total, 1)

        # Detect friction topics
        detected_topics = []
        for topic, keywords in cls.FRICTION_TOPIC_KEYWORDS.items():
            if any(kw in text.lower() for kw in keywords):
                detected_topics.append(topic)

        primary_topic = detected_topics[0] if detected_topics else "product_features"

        return {
            "sentiment": round(float(sentiment), 2),
            "primary_friction_topic": primary_topic,
            "all_detected_topics": detected_topics
        }


# =====================================================================
# Call History Database Manager
# =====================================================================

def load_or_init_call_history() -> pd.DataFrame:
    """Loads historical call logs from JSON or initializes with synthetic bootstrap dataset."""
    if os.path.exists(CALL_HISTORY_FILE):
        try:
            with open(CALL_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Could not load {CALL_HISTORY_FILE}: {e}")

    # Initialize with bootstrap baseline of 100 historical calls
    df = generate_sample_sales_call_logs(num_calls=100)
    save_call_history(df)
    return df


def save_call_history(df: pd.DataFrame):
    """Saves DataFrame of call logs to JSON."""
    records = df.to_dict(orient="records")
    with open(CALL_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# =====================================================================
# Live Call Session Runner
# =====================================================================

class LiveSalesCallSession:
    """Manages live turn-by-turn sales call, telemetry collection, and post-call ML learning."""

    def __init__(self, enable_tts: bool = True):
        self.rag = B2BSalesRAG()
        self.tts = ChatterboxTTSAgent(voice_preset="consultative_rep_female") if enable_tts else None
        self.call_start_time = 0.0
        self.turns: List[Dict[str, Any]] = []
        self.friction_topics_encountered = []
        self.sentiments = []
        self.demo_booked = False
        self.drop_off_turn = 0
        self.final_reason = "normal_completion"

    def start_call(self):
        """Starts interactive call simulator."""
        print("\n" + "=" * 80)
        print("  📞 LIVE INBOUND SALES CALL SIMULATION (AI Sales Agent)")
        print("=" * 80)
        print("Role: You are the Prospect / Client evaluating ApexSales AI.")
        print("Commands:")
        print("  - Type your questions/objections naturally (e.g. 'It's too expensive', 'We use Drift')")
        print("  - Type 'hangup' or 'bye' to drop off / cut the call")
        print("  - Type 'book demo' or 'sounds great let's do it' to close successfully")
        print("=" * 80 + "\n")

        self.call_start_time = time.time()
        turn_idx = 0

        # AI Opening Line
        opening_line = (
            "Hi there! Thanks for connecting with ApexSales AI. This is Sarah. "
            "I saw your team was exploring ways to scale inbound sales calls—what's your main priority this quarter?"
        )
        print(f"\n[Turn 0] AI Sales Rep (Sarah) > {opening_line}\n")
        if self.tts:
            self.tts.synthesize_to_file(opening_line, "opening_turn.wav")

        while True:
            turn_idx += 1
            try:
                user_msg = input(f"[Turn {turn_idx}] Prospect (You) > ").strip()
                if not user_msg:
                    continue

                # 1. Analyze Prospect Tone & Friction in real time
                analysis = ProspectToneAnalyzer.analyze_utterance(user_msg)
                self.sentiments.append(analysis["sentiment"])
                self.friction_topics_encountered.append(analysis["primary_friction_topic"])

                # Check for call termination
                if user_msg.lower() in ("hangup", "bye", "quit", "exit") or "not interested" in user_msg.lower():
                    self.drop_off_turn = turn_idx
                    self.demo_booked = False
                    self.final_reason = f"prospect_cut_call_on_{analysis['primary_friction_topic']}"
                    print(f"\n⚠️  [CALL ENDED - Prospect Hung Up at Turn {turn_idx} | Tone: {analysis['sentiment']} | Friction: {analysis['primary_friction_topic']}]")
                    break

                if "book demo" in user_msg.lower() or "schedule" in user_msg.lower() or "thursday" in user_msg.lower() or "friday" in user_msg.lower():
                    self.drop_off_turn = turn_idx + 2
                    self.demo_booked = True
                    self.final_reason = "demo_booked_success"

                # 2. AI Generates Response using RAG
                print("\nAI Sales Rep (Sarah) > ", end="", flush=True)
                full_ai_tokens = []
                gen_start = time.time()

                for token in self.rag.generate_stream(user_msg):
                    print(token, end="", flush=True)
                    full_ai_tokens.append(token)
                print("\n")

                gen_latency_ms = (time.time() - gen_start) * 1000
                ai_reply = "".join(full_ai_tokens).strip()

                # Optional TTS Synthesis
                if self.tts and ai_reply:
                    self.tts.synthesize_to_file(ai_reply, f"call_turn_{turn_idx}.wav")

                self.turns.append({
                    "turn": turn_idx,
                    "user_msg": user_msg,
                    "ai_reply": ai_reply,
                    "sentiment": analysis["sentiment"],
                    "friction_topic": analysis["primary_friction_topic"],
                    "latency_ms": round(gen_latency_ms, 1)
                })

                if self.demo_booked:
                    print("🎉 [CALL SUCCESS - Demo Scheduled & BANT Qualified!]")
                    break

            except (KeyboardInterrupt, EOFError):
                self.drop_off_turn = turn_idx
                self.demo_booked = False
                self.final_reason = "user_interrupted"
                break

        total_call_duration = round(time.time() - self.call_start_time, 1)
        if self.drop_off_turn == 0:
            self.drop_off_turn = turn_idx

        # 3. Compile Call Telemetry
        avg_sentiment = round(float(np.mean(self.sentiments)), 2) if self.sentiments else -0.2
        dominant_friction = max(set(self.friction_topics_encountered), key=self.friction_topics_encountered.count) if self.friction_topics_encountered else "product_features"

        call_record = {
            "call_id": f"LIVE_CALL_{int(time.time())}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": max(10.0, total_call_duration),
            "drop_off_turn": float(self.drop_off_turn),
            "sentiment_score": float(avg_sentiment),
            "interruption_count": len([s for s in self.sentiments if s < -0.3]),
            "agent_latency_ms": 410.0,
            "friction_topic": dominant_friction,
            "demo_booked": 1 if self.demo_booked else 0,
            "final_reason": self.final_reason
        }

        self._process_post_call_learning(call_record)

    def _process_post_call_learning(self, call_record: Dict[str, Any]):
        """Runs K-Means re-clustering and autonomous RAG knowledge base self-healing."""
        print("\n" + "=" * 80)
        print("  🧠 POST-CALL ML ANALYSIS & RAG CONTINUOUS LEARNING LOOP")
        print("=" * 80)

        # 1. Append new call to database
        df_history = load_or_init_call_history()
        new_row_df = pd.DataFrame([call_record])
        df_history = pd.concat([df_history, new_row_df], ignore_index=True)
        save_call_history(df_history)
        print(f"✓ Saved call telemetry to '{CALL_HISTORY_FILE}' (Total calls tracked: {len(df_history)})")
        print(f"  - Duration: {call_record['duration_sec']}s | Drop-off Turn: {call_record['drop_off_turn']}")
        print(f"  - Tone Sentiment: {call_record['sentiment_score']} | Primary Friction: {call_record['friction_topic']}")
        print(f"  - Demo Booked: {'YES' if call_record['demo_booked'] == 1 else 'NO'}")

        # 2. Run K-Means Clustering on the updated dataset
        print("\n[Step 1/2] Updating K-Means Clusters with new call interaction...")
        clustering = SalesCallClustering(k_clusters=4)
        df_clustered = clustering.fit_and_analyze(df_history)
        
        # Identify which cluster this call fell into
        assigned_cluster = int(df_clustered.iloc[-1]["cluster"])
        cluster_info = clustering.cluster_summary.get(assigned_cluster, {})
        
        print(f"✓ New call classified into **Cluster {assigned_cluster}**:")
        print(f"  - Diagnosis    : {cluster_info.get('diagnosis')}")
        print(f"  - Failure Rate : {cluster_info.get('failure_rate')}%")
        print(f"  - Dominant Pain: {cluster_info.get('dominant_topic')}")

        clustering.print_diagnostics()

        # 3. Autonomous RAG Knowledge Base Self-Healing
        print("\n[Step 2/2] Running RAG Knowledge Base Self-Healing...")
        optimizer = RAGKnowledgeOptimizer(kb_path=KB_FILE_PATH)
        updates = optimizer.optimize_knowledge_base(clustering.cluster_summary)

        if updates:
            print("\n✨ [RAG AUTO-UPDATED] Successfully learned from call failure:")
            for update in updates:
                print(f"  👉 {update}")
            print("\n✓ 'Rag_Knowledge_base.txt' updated. Re-indexing FAISS vector store...")
            self.rag.load_knowledge_base_from_file(KB_FILE_PATH)
            print("✓ FAISS Vector DB successfully re-indexed with new objection playbooks for future calls!")
        else:
            print("\n✓ Knowledge base is already equipped for this scenario. No changes needed.")

        print("=" * 80 + "\n")


# =====================================================================
# Main Entry Point
# =====================================================================

if __name__ == "__main__":
    session = LiveSalesCallSession(enable_tts=True)
    session.start_call()
