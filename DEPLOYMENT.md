# 🚀 Render Deployment & MongoDB Atlas Setup Guide — ApexSales AI

This guide explains how to deploy **ApexSales AI** as a unified full-stack application entirely on **Render** (Frontend UI + WebSocket Audio Gateway + LangGraph Reasoning + ChromaDB RAG), and how to connect **MongoDB Atlas** seamlessly.

---

## 🏗️ Unified Render Architecture
```text
[ Browser (Chrome / Edge / Firefox) ]
        │
        │ HTTPS (Web UI & REST APIs) + WSS (Real-Time Audio Streams)
        ▼
[ Render Web Service: real-time-ai-b2b-sales-agent ]
  ├── GET / (Unified Sales Console Web UI)
  ├── WS /ws/audio (AssemblyAI 16kHz Real-Time STT Gateway)
  ├── POST /api/chat (LangGraph Multi-Step Reasoning Engine)
  ├── GET /api/rag/query (Persistent ChromaDB Vector Knowledge Store)
  ├── Edge-TTS Neural Voice Synthesis (Jenny / Guy)
  └── Telemetry Node -> Live MongoDB Atlas (`apexsales_db`)
```

---

## Part 1: Fix MongoDB Atlas Connection

If MongoDB Atlas is rejecting connections with `[SSL: TLSV1_ALERT_INTERNAL_ERROR]`, it is because **the incoming IP address is blocked in Atlas Network Access**.

Follow these exact steps:

1. Log into [MongoDB Atlas](https://cloud.mongodb.com/).
2. Select your Project and Cluster (`Cluster0`).
3. In the left navigation menu, click **Network Access** (under **Security**).
4. Click the green **+ Add IP Address** button.
5. Click **Allow Access From Anywhere** (this automatically sets IP `0.0.0.0/0`).
6. Click **Confirm**.
7. Wait 30–60 seconds until the status shows **Active**.

### Verify Database User Credentials
1. In the left menu, click **Database Access**.
2. Ensure user `kshriday_db_user` exists and has **Read and write to any database** privileges.
3. If you ever update the password, update it in Render and `.env`.

---

## Part 2: Deploy to Render (Unified Full-Stack)

Render hosts both your frontend UI and backend services together on a single URL:
**`https://real-time-ai-b2b-sales-agent.onrender.com/`**

### Step 1: Render Web Service Configuration
- **Repository:** `HridaySharma2002/Real-Time-AI-B2B-Sales-Agent`
- **Environment:** `Python`
- **Region:** `Oregon (US West)`
- **Branch:** `main`
- **Build Command:**
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command:**
  ```bash
  python -m uvicorn main:app --app-dir agentic-dialogue-engine --host 0.0.0.0 --port $PORT
  ```
- **Health Check Path:**
  ```text
  /api/health
  ```

### Step 2: Configure Environment Variables in Render Dashboard
Go to your Render Web Service -> **Environment** tab and ensure the following variables are present:

| Key | Recommended Value |
|---|---|
| `PYTHON_VERSION` | `3.11.9` |
| `ASSEMBLYAI_API_KEY` | `your_assemblyai_api_key_from_env` |
| `GROQ_API_KEY` | `your_groq_api_key_from_env` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` |
| `MONGODB_URI` | `mongodb+srv://<username>:<password>@cluster0.1nsmy0e.mongodb.net/apexsales_db?retryWrites=true&w=majority&appName=Cluster0` |
| `MONGODB_USERNAME` | `<your_mongodb_username>` |
| `MONGODB_PASSWORD` | `<your_mongodb_password>` |

---

## Part 3: Live Verification & Testing

Once Render completes deploying:

1. Open your live application in your browser:
   **`https://real-time-ai-b2b-sales-agent.onrender.com/`**
2. You will see the **ApexSales AI Interactive Sales Console**.
3. Check status indicators:
   - **WebSocket:** Connected (Green dot)
   - **Mic:** Ready
4. Check system health and MongoDB connectivity:
   **`https://real-time-ai-b2b-sales-agent.onrender.com/api/health`**
   ```json
   {
     "status": "healthy",
     "mongodb_connected": true,
     "platform": "ApexSales AI Real-Time B2B Sales Agent"
   }
   ```
5. Test live audio:
   - Click **"Test Sample Audio"** to stream 16kHz PCM audio without a microphone.
   - Click **"Start Real-Time Call"** to talk with the AI sales agent live via your microphone.
