# ApexSales AI - Real-Time AI B2B Sales Agent

An enterprise-grade, real-time speech-to-speech AI B2B Sales Agent platform built with **FastAPI**, **LangGraph**, **ChromaDB**, **AssemblyAI Streaming STT**, **Chatterbox/Kokoro Neural TTS**, and a **Reactive Java Spring WebFlux Microservice**.

---

## 🏛️ System Architecture

```
[Browser Client (Web Audio API + MediaRecorder)]
       │ (16kHz 16-bit PCM Audio Stream over WebSocket)
       ▼
[Python FastAPI Gateway (/ws/audio)]
       │
       ├──▶ [AssemblyAI Streaming STT (v3)] ──▶ Real-time Transcripts (Partial & Final)
       │
       ▼
[LangGraph Multi-Step Reasoning Agent]
       ├── [1. K-Means Lead Segmentation] ──▶ Evaluates call metrics & classifies persona
       ├── [2. Dynamic ChromaDB RAG Pipeline] ──▶ Sub-50ms vector search across knowledge base
       ├── [3. Groq LLM (openai/gpt-oss-120b)] ──▶ Low-latency persuasive sales reasoning
       └── [4. Chatterbox / Kokoro TTS] ────────▶ In-memory audio synthesis
       │
       ▼ (Base64 WAV Audio + Telemetry over WebSocket)
[Browser Client (Web Audio API Playback & Waveform Visualizer)]
       │
       ▼ (Non-blocking Async Data Sync)
[Java Spring WebFlux Microservice (:8080)]
       │ (Reactive WebClient & Data Persistence)
       ▼
[Redis Cache (:6379)] + [MongoDB Atlas Database]
```

---

## 🚀 Key Modules & Capabilities

1. **Client Side (Web Browser Console)**:
   - **`agentic-dialogue-engine/test_ui.html`**: Modern, reactive sales console.
   - Microphone streaming using `MediaRecorder` & `ScriptProcessor` at 16kHz PCM.
   - Real-time audio playback using the `Web Audio API` decoding neural voice responses.
   - Live Canvas Audio Waveform Visualizer tracking speaker and prospect sound waves.
   - Prospect Intelligence Panel displaying K-Means persona, sentiment polarity, and RAG context cards.

2. **Real-Time Processing (Python FastAPI Backend)**:
   - **`agentic-dialogue-engine/main.py`**: High-performance ASGI WebSocket and REST gateway.
   - **`agentic-dialogue-engine/stt_service.py`**: AssemblyAI v3 streaming speech-to-text.
   - **`agentic-dialogue-engine/agent_workflow.py`**: LangGraph StateGraph agent executing lead segmentation, ChromaDB RAG retrieval, and Groq LLM generation.
   - **`services/chroma_rag.py`**: Persistent ChromaDB vector store indexing `Rag_Knowledge_base.txt`.
   - **`services/redis_cache.py`**: Sub-5ms caching for dialogue sessions with in-memory TTL failover.
   - **`services/mongo_service.py`**: Lead profiles and interaction history persistence with local JSON failover.
   - **`services/b2b_enrichment.py`**: External B2B firmographics and tech stack inference.
   - **`services/spring_sync_client.py`**: Async HTTP client syncing call turns to Spring WebFlux.

3. **Async Microservices (Java Spring WebFlux)**:
   - **`spring-microservice/`**: Reactive non-blocking microservice built on Spring Boot 3 & Project Reactor.
   - Exposes reactive endpoints:
     - `POST /api/leads` - Record or update lead
     - `GET /api/leads` - Non-blocking stream of all leads
     - `POST /api/leads/{id}/interactions` - Append call dialogue turn
     - `POST /api/analytics` - Persist post-call analytics & K-Means failure diagnostics
     - `GET /api/health` - Health status
   - `FastApiWebClient`: Reactive `WebClient` communicating with FastAPI.

---

## 🛠️ Setup & Running

### 1. Configure `.env`
Ensure your `.env` contains:
```env
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_API_KEY=your_gemini_api_key

# Optional / External services
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.mongodb.net/apexsales_db
REDIS_HOST=localhost
REDIS_PORT=6379
SPRING_SERVICE_URL=http://localhost:8080
```

### 2. Start the Java Spring WebFlux Microservice
```bash
cd spring-microservice
mvn clean compile
java -jar target/sales-microservice-1.0.0.jar
```
*Runs on `http://localhost:8080`.*

### 3. Start the FastAPI Real-Time Audio Server
```bash
cd agentic-dialogue-engine
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
*Runs on `http://127.0.0.1:8000`.*

### 4. Interactive Browser UI
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** or **[http://127.0.0.1:8000/test](http://127.0.0.1:8000/test)** in your browser:
- Deployed Live on Render: **[https://real-time-ai-b2b-sales-agent.onrender.com/](https://real-time-ai-b2b-sales-agent.onrender.com/)**
- Click **"Start Real-Time Call"** to speak directly with the AI sales agent via microphone.
- Or click **"Test Sample Audio"** to run a simulated audio stream without a microphone.

### 5. Automated End-to-End WebSocket Test
```bash
cd agentic-dialogue-engine
python test_client.py
```
