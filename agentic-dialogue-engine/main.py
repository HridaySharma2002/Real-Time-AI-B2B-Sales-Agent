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

# Ensure root workspace is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

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
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        html_file = os.path.join(os.path.dirname(__file__), "test_ui.html")
        if os.path.exists(html_file):
            with open(html_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return JSONResponse(content={
        "status": "running",
        "platform": "ApexSales AI Real-Time B2B Sales Agent",
        "version": "2.0.0",
        "endpoints": {
            "sales_console_ui": "/test",
            "audio_websocket": "/ws/audio",
            "text_chat_api": "/api/chat",
            "leads_api": "/api/leads",
            "enrichment_api": "/api/b2b/enrich",
            "rag_query_api": "/api/rag/query",
            "analytics_api": "/api/analytics"
        }
    })


@app.get("/test", response_class=HTMLResponse)
def get_test_page():
    html_file = os.path.join(os.path.dirname(__file__), "test_ui.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Test UI file not found</h1>"


@app.get("/sample-audio")
def get_sample_audio():
    sample_file = os.path.join(os.path.dirname(__file__), "test_sample_16k.wav")
    if os.path.exists(sample_file):
        return FileResponse(sample_file, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"error": "Sample audio file not found"})


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Direct text chat endpoint running full LangGraph reasoning + TTS."""
    result = agent_app.invoke({
        "messages": [HumanMessage(content=req.message)],
        "session_id": req.session_id,
        "lead_id": req.lead_id
    })

    reply_text = result["messages"][-1].content
    audio_base64 = None

    if req.synthesize_audio:
        tts = _get_tts_agent()
        if tts:
            try:
                wav_bytes = tts.synthesize_to_wav_bytes(reply_text)
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
    
    try:
        transcriber.connect()
        logger.info(f"AssemblyAI Transcriber connected for WebSocket session: {session_id}")
        await websocket.send_json({
            "type": "transcriber_ready",
            "session_id": session_id,
            "message": "AI Speech Recognizer Ready"
        })
    except Exception as e:
        logger.error(f"Failed to connect AssemblyAI Transcriber: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"STT Connection Error: {str(e)}"
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
                    if cmd.get("type") in ("end_turn", "silence"):
                        transcriber.stream(b"\x00" * 32000)
                except Exception:
                    pass
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        transcriber.close()
        logger.info(f"Transcriber session closed: {session_id}")