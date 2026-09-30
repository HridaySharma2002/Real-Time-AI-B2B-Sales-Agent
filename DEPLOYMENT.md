# 🚀 Deployment Guide — ApexSales AI

This guide explains how to deploy **ApexSales AI** across **Render** (Backend FastAPI + AI Agents) and **Netlify** (Edge Royal Light Mode Frontend).

---

## 🏗️ Architecture Overview
```text
[ Browser / Netlify Frontend ]
        │  (WebSocket + REST)
        ▼
[ Render Web Service (FastAPI) ]
  ├── AssemblyAI Real-Time STT
  ├── LangGraph Agent (Groq / Gemini)
  ├── Edge-TTS Neural Voice (Jenny / Guy)
  ├── ChromaDB RAG Knowledge Vector Store
  └── Non-Blocking Telemetry (Redis / MongoDB / Spring)
```

---

## Part 1: Deploy Backend to Render

### Option A: Via Blueprint (`render.yaml`)
1. Push this repository to your GitHub account.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect this GitHub repository. Render will automatically detect [`render.yaml`](file:///C:/Users/USER/Desktop/AI-B2B-Sales_Agent/Real-Time-AI-B2B-Sales-Agent/render.yaml).
5. In the Environment Variables section, fill in your secret keys:
   - `ASSEMBLYAI_API_KEY`: Your AssemblyAI API key
   - `GROQ_API_KEY`: Your Groq API key
   - `MONGODB_URI`: Your MongoDB Atlas connection URI
   - `MONGODB_USERNAME`: Your MongoDB username
   - `MONGODB_PASSWORD`: Your MongoDB password
6. Click **Apply**. Render will install dependencies and start the uvicorn server.
7. Once deployed, copy your backend URL:
   `https://<your-app-name>.onrender.com`

### Option B: Manual Web Service
- **Runtime:** Python
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python -m uvicorn main:app --app-dir agentic-dialogue-engine --host 0.0.0.0 --port $PORT`
- **Environment Variables:** Add keys from `.env.example`

---

## Part 2: Deploy Frontend to Netlify

### Option A: Netlify Git Integration
1. Log into [Netlify](https://app.netlify.com/).
2. Click **Add new site** → **Import an existing project**.
3. Connect your GitHub repository.
4. Set the build settings:
   - **Base directory:** Leave blank
   - **Package directory / Publish directory:** `frontend`
5. Click **Deploy Site**.
6. Netlify will deploy your site in ~10 seconds. You will get a URL like `https://<your-site>.netlify.app`.

### Option B: Netlify Drop (Zero Configuration)
1. In your local workspace, locate the [`frontend`](file:///C:/Users/USER/Desktop/AI-B2B-Sales_Agent/Real-Time-AI-B2B-Sales-Agent/frontend) directory.
2. Go to [Netlify Drop](https://app.netlify.com/drop).
3. Drag and drop the `frontend` folder directly into the browser.
4. Your UI will be live instantly!

---

## Part 3: Connect Frontend to Backend

1. Open your deployed Netlify frontend in your browser.
2. Click the **"⚙️ Cloud Server"** button in the top navigation bar.
3. Enter your deployed Render backend URL:
   `https://<your-backend-name>.onrender.com`
4. Click **Save & Connect**.
5. The frontend will automatically convert `https://` into `wss://` for real-time audio streaming, and remember your choice across browser reloads via `localStorage`.
6. Click **Start Real-Time Call** and speak through your microphone!

---

## Part 4: Local Development & Testing

Run backend locally:
```powershell
python -m uvicorn main:app --app-dir agentic-dialogue-engine --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000/test`** to view the live Edge Royal Light Mode UI directly from your local server.
