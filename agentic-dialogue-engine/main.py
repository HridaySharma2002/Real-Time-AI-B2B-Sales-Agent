"""
agentic-dialogue-engine/main.py - Real-Time FastAPI Server & Streaming WebSocket Gateway
Part of ApexSales AI Real-Time B2B Sales Agent.

Features:
- Real-time audio WebSocket endpoint (`/ws/audio`):
  * Ingests 16kHz 16-bit PCM audio stream from browser microphone.
  * Streams to AssemblyAI Streaming STT.
  * Triggers LangGraph Multi-Step Reasoning Agent (K-Means + ChromaDB RAG + Groq/Gemini).
  * Synthesizes audio using Chatterbox / Kokoro TTS.
  * Returns live transcripts, classified persona, RAG context, and base64 audio to browser.
- REST Endpoints:
  * `/api/chat`: Direct text conversation endpoint.
  * `/api/leads`: Create/list B2B sales leads.
  * `/api/b2b/enrich`: External B2B firmographic enrichment.
  * `/api/rag/query`: Dynamic ChromaDB vector search.
  * `/api/analytics`: Real-time call failure diagnostics and K-Means clusters.
  * `/test` and `/`: Interactive Browser Sales Console Web UI.
"""

import asyncio
import os
import sys
import base64
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from dotenv import load_dotenv

# Ensure root workspace is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

load_dotenv(os.path.join(root_dir, ".env"))
load_dotenv()

from stt_service import get_transcriber, _get_tts_agent
from agent_workflow import agent_app
from langchain_core.messages import HumanMessage
from services.chroma_rag import get_chroma_rag
from services.redis_cache import get_redis_cache
from services.mongo_service import get_mongo_service
from services.b2b_enrichment import B2BEnrichmentService
from services.spring_sync_client import get_spring_sync_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SalesAgentServer")

app = FastAPI(
    title="ApexSales AI Real-Time B2B Sales Agent",
    version="2.0.0",
    description="Real-Time Speech-to-Speech B2B Sales Dialogue Engine with LangGraph, ChromaDB, and K-Means"
)

# Enable CORS for external frontends and microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request / Response Schemas
# =====================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "web_session"
    lead_id: Optional[str] = "lead_web"
    synthesize_audio: Optional[bool] = True


class LeadCreateRequest(BaseModel):
    company: str
    contact_name: Optional[str] = "Decision Maker"
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


# =====================================================================
# REST Endpoints
# =====================================================================

@app.get("/")
def read_root(request: Request):
    """Serves the Unified Interactive Sales Console Web UI."""
    # Check frontend/index.html first (Unified Single-Deployment)
    frontend_html = os.path.join(root_dir, "frontend", "index.html")
    if os.path.exists(frontend_html):
        with open(frontend_html, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    # Fallback to local test_ui.html
    html_file = os.path.join(os.path.dirname(__file__), "test_ui.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ApexSales AI Real-Time B2B Sales Agent</h1><p>Web UI file not found.</p>")


@app.get("/test", response_class=HTMLResponse)
def get_test_page(request: Request):
    """Direct alias to the Interactive Sales Console Web UI."""
    return read_root(request)


@app.get("/healthz")
@app.get("/api/health")
def get_health_status():
    """Health check endpoint for Render and infrastructure monitoring."""
    db = get_mongo_service()
    cache = get_redis_cache()
    return JSONResponse(content={
        "status": "healthy",
        "platform": "ApexSales AI Real-Time B2B Sales Agent",
        "version": "2.0.0",
        "mongodb_connected": db.is_connected(),
        "redis_connected": cache.is_available(),
        "endpoints": {
            "sales_console_ui": "/",
            "audio_websocket": "/ws/audio",
            "text_chat_api": "/api/chat",
            "leads_api": "/api/leads",
            "enrichment_api": "/api/b2b/enrich",
            "rag_query_api": "/api/rag/query",
            "analytics_api": "/api/analytics"
        }
    })


@app.get("/sample-audio")
def get_sample_audio():
    sample_file = os.path.join(os.path.dirname(__file__), "test_sample_16k.wav")
    if os.path.exists(sample_file):
        return FileResponse(sample_file, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"error": "Sample audio file not found"})


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Direct text chat endpoint running full LangGraph reasoning + TTS non-blocking."""
    def _run_agent():
        return agent_app.invoke({
            "messages": [HumanMessage(content=req.message)],
            "session_id": req.session_id,
            "lead_id": req.lead_id
        })

    result = await asyncio.to_thread(_run_agent)
    reply_text = result["messages"][-1].content
    audio_base64 = None

    if req.synthesize_audio:
        tts = _get_tts_agent()
        if tts:
            try:
                wav_bytes = await asyncio.to_thread(tts.synthesize_to_wav_bytes, reply_text)
                if wav_bytes:
                    audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
            except Exception as e:
                logger.warning(f"Chat TTS synthesis error: {e}")

    return {
        "reply": reply_text,
        "persona": result.get("persona"),
        "cluster_id": result.get("cluster_id"),
        "friction_topic": result.get("friction_topic"),
        "rag_context": result.get("rag_context"),
        "enrichment": result.get("enrichment"),
        "audio_base64": audio_base64
    }


@app.get("/api/leads")
def list_leads():
    db = get_mongo_service()
    return {"leads": db.get_recent_leads(limit=25)}


@app.post("/api/leads")
def create_lead(req: LeadCreateRequest):
    db = get_mongo_service()
    lead_id = db.save_lead(req.dict())
    # Async sync to Java Spring WebFlux
    spring_client = get_spring_sync_client()
    spring_client.dispatch_sync_lead_bg({"lead_id": lead_id, **req.dict()})
    return {"status": "success", "lead_id": lead_id}


@app.get("/api/b2b/enrich")
def enrich_company(query: str = Query(..., description="Company name or domain to enrich")):
    return B2BEnrichmentService.enrich_from_text(query)


@app.get("/api/rag/query")
def query_rag(q: str = Query(..., description="Query knowledge base")):
    rag = get_chroma_rag()
    results = rag.query(q, n_results=3)
    return {"query": q, "results": results}


@app.get("/api/analytics")
def get_analytics():
    """Returns K-Means clustering telemetry and call diagnostics."""
    db = get_mongo_service()
    leads = db.get_recent_leads(50)
    return {
        "total_leads_tracked": len(leads),
        "cluster_diagnostics": {
            0: {"persona": "Pricing-Sensitive Evaluator", "tactic": "ROI Reframe & Pilot POC"},
            1: {"persona": "High-Converting Enterprise Champion", "tactic": "Custom Integrations & Demo Close"},
            2: {"persona": "Early Gatekeeper", "tactic": "High-Impact 1-Liner Value Prop"},
            3: {"persona": "Technical Solutions Architect", "tactic": "Architecture, SLAs & SOC2 Compliance"}
        },
        "recent_leads": leads[:5]
    }


# =====================================================================
# Real-Time Audio Streaming WebSocket Endpoint
# =====================================================================

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    session_id = f"call_{int(asyncio.get_event_loop().time() * 1000)}"
    lead_id = f"lead_{int(asyncio.get_event_loop().time())}"

    # Notify client of connected session
    await websocket.send_json({
        "type": "connection_established",
        "session_id": session_id,
        "lead_id": lead_id,
        "message": "Connected to ApexSales AI Real-Time Audio Engine"
    })

    def send_transcript(text: str, is_final: bool):
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "is_final": is_final
                }),
                loop
            )
        except Exception as e:
            logger.error(f"Error sending transcript: {e}")

    def send_agent_response(payload: Any):
        try:
            msg = {"type": "agent_response"}
            if isinstance(payload, dict):
                msg.update(payload)
            else:
                msg["text"] = str(payload)

            asyncio.run_coroutine_threadsafe(
                websocket.send_json(msg),
                loop
            )
        except Exception as e:
            logger.error(f"Error sending agent response payload: {e}")

    transcriber = get_transcriber(
        on_transcript=send_transcript,
        on_agent_response=send_agent_response,
        sample_rate=16000,
        session_id=session_id,
        lead_id=lead_id
    )
    
    stt_connected = False
    try:
        transcriber.connect()
        stt_connected = True
        logger.info(f"AssemblyAI Transcriber connected for WebSocket session: {session_id}")
        await websocket.send_json({
            "type": "transcriber_ready",
            "session_id": session_id,
            "message": "AI Speech Recognizer Ready"
        })
    except Exception as e:
        logger.warning(f"AssemblyAI Transcriber unavailable ({e}). WebSocket remains active for text and playback.")
        await websocket.send_json({
            "type": "stt_status",
            "connected": False,
            "message": "Real-time speech recognition is in standby. Text chat and sample audio remain operational."
        })

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                transcriber.stream(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    import json
                    cmd = json.loads(msg["text"])
                    cmd_type = cmd.get("type")
                    if cmd_type in ("end_turn", "silence"):
                        transcriber.stream(b"\x00" * 32000)
                    elif cmd_type == "start_call":
                        if not transcriber._connected:
                            try:
                                transcriber.connect()
                                await websocket.send_json({
                                    "type": "transcriber_ready",
                                    "session_id": session_id,
                                    "message": "AI Speech Recognizer Ready"
                                })
                            except Exception as conn_err:
                                logger.warning(f"Could not reconnect on start_call: {conn_err}")
                    elif cmd_type == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    pass
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        try:
            transcriber.close()
        except Exception:
            pass
        logger.info(f"Transcriber session closed: {session_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)