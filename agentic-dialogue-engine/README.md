# Agentic Dialogue Engine

Real-time Speech-to-Speech / Speech-to-Text dialogue agent for B2B Sales, powered by **AssemblyAI Streaming (v3)**, **LangGraph**, and **Groq**.

## Architecture & Flow

```
[Customer Mic / Audio]
        │ (16kHz 16-bit PCM Audio Stream)
        ▼
[FastAPI WebSocket (/ws/audio)]
        │
        ▼
[AssemblyAI Streaming v3 STT] ──▶ Real-time transcription (turns)
        │
        ▼ (Final transcript)
[LangGraph Dialogue Agent] ─────▶ Sales Persona + RAG Context
        │
        ▼
[Groq LLM (openai/gpt-oss-120b)] ▶ Concise, low-latency sales response
        │
        ▼
[FastAPI WebSocket] ───────────▶ Real-time response to frontend & TTS
```

## Setup & Running

### 1. Configure Environment
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```env
GROQ_API_KEY=your_groq_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Server
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Testing

### Interactive Browser UI
Open **[http://127.0.0.1:8000/test](http://127.0.0.1:8000/test)** in your browser:
* Click **"Start Talking (Mic)"** to speak directly into your microphone.
* Or click **"Stream Sample Audio"** to run a simulated audio stream without a mic.

### Automated Terminal Test
Run the end-to-end WebSocket test client:
```bash
python test_client.py
```

### Standalone Direct Pipeline Test
Test AssemblyAI and LangGraph directly without running FastAPI:
```bash
python test_audio_flow.py
```
