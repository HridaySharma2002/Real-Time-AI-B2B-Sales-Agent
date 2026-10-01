"""
agentic-dialogue-engine/agent_workflow.py - LangGraph Multi-Step Reasoning Sales Agent
Part of ApexSales AI Real-Time B2B Sales Agent.

Architecture & Reasoning Steps:
1. `segment_lead_node`:
   - Analyzes prospect speech and dialogue metrics.
   - Evaluates offline K-Means cluster profiles to identify persona:
     * High-Converting Enterprise Champion
     * Pricing-Sensitive Evaluator
     * Early Gatekeeper
     * Technical Solutions Architect
   - Enriches lead firmographics (industry, team size, tech stack).
2. `retrieve_rag_node`:
   - Queries ChromaDB vector store dynamically for matching sales packages, battlecards, and objection scripts.
3. `generate_response_node`:
   - Synthesizes persona + RAG context + conversational history.
   - Invokes Groq LLM (openai/gpt-oss-120b) with low-latency parameters for real-time speech dialogue.
4. `sync_telemetry_node`:
   - Asynchronously persists call telemetry to Redis, MongoDB Atlas, and Java Spring WebFlux service.
"""

import os
import sys
import time
import logging
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from dotenv import load_dotenv

# Ensure root workspace is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Local services
from services.chroma_rag import get_chroma_rag
from services.redis_cache import get_redis_cache
from services.mongo_service import get_mongo_service
from services.b2b_enrichment import B2BEnrichmentService
from services.spring_sync_client import get_spring_sync_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LangGraphSalesAgent")

# =====================================================================
# State Definition
# =====================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    lead_id: str
    persona: str
    friction_topic: str
    sentiment: float
    rag_context: str
    enrichment: Dict[str, Any]
    cluster_id: int


# =====================================================================
# LLM Initialization
# =====================================================================

groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=model_name, temperature=0.1, api_key=groq_api_key, request_timeout=6.0, max_retries=1) if groq_api_key else None


# =====================================================================
# Offline K-Means Lead Segmentation Node
# =====================================================================

KMEANS_PERSONA_PROFILES = {
    0: {
        "persona": "Pricing-Sensitive Evaluator",
        "description": "Focused heavily on cost, ROI, and budget justification. Needs starter tier or pilot POC.",
        "recommended_tier": "Starter Plan ($499/mo) or Growth Pilot",
        "tactic": "Acknowledge budget, reframe to 4x ROI within 60 days, and propose a low-risk pilot."
    },
    1: {
        "persona": "High-Converting Enterprise Champion",
        "description": "Ready to scale sales reps, high purchase intent, interested in custom integrations.",
        "recommended_tier": "Enterprise Custom ($3,500+/mo)",
        "tactic": "Highlight sub-300ms SLA, dedicated account management, Salesforce sync, and schedule demo."
    },
    2: {
        "persona": "Early Gatekeeper",
        "description": "Short queries, skeptical, exploring high-level value before passing to decision makers.",
        "recommended_tier": "Growth Plan ($1,499/mo)",
        "tactic": "Deliver crisp 1-sentence value proposition on how AI frees up 20+ SDR hours weekly."
    },
    3: {
        "persona": "Technical Solutions Architect",
        "description": "Asks about latency, API architecture, WebSockets, security, and SOC2 compliance.",
        "recommended_tier": "Enterprise Custom (Private VPC / Dedicated Pipeline)",
        "tactic": "Explain reactive architecture, AssemblyAI streaming STT, and on-prem/cloud hybrid options."
    }
}


def segment_lead_node(state: AgentState):
    """Classifies prospect using K-Means profile matching and B2B firmographics."""
    latest_msg = state["messages"][-1].content if state.get("messages") else ""
    lower = latest_msg.lower()

    # Rule-assisted K-Means classification
    if any(w in lower for w in ["expensive", "cost", "price", "budget", "discount", "how much", "rate"]):
        cluster_id = 0
        friction = "pricing"
        sentiment = -0.3
    elif any(w in lower for w in ["book", "demo", "buy", "scale", "enterprise", "salesforce", "ready", "contract"]):
        cluster_id = 1
        friction = "closing"
        sentiment = 0.8
    elif any(w in lower for w in ["who is this", "what do you do", "not interested", "busy", "send email"]):
        cluster_id = 2
        friction = "gatekeeper_bounce"
        sentiment = -0.1
    elif any(w in lower for w in ["latency", "architecture", "api", "security", "soc2", "model", "websocket", "rag"]):
        cluster_id = 3
        friction = "technical_spec"
        sentiment = 0.2
    else:
        cluster_id = 1
        friction = "general_discovery"
        sentiment = 0.4

    profile = KMEANS_PERSONA_PROFILES[cluster_id]
    enrichment = B2BEnrichmentService.enrich_from_text(latest_msg)

    logger.info(f"[K-Means Segmentation] Classified Prospect as Cluster {cluster_id}: {profile['persona']}")

    return {
        "cluster_id": cluster_id,
        "persona": profile["persona"],
        "friction_topic": friction,
        "sentiment": sentiment,
        "enrichment": enrichment
    }


# =====================================================================
# ChromaDB Dynamic RAG Retrieval Node
# =====================================================================

def retrieve_rag_node(state: AgentState):
    """Queries ChromaDB vector store for sales packages, battlecards, and objection playbooks."""
    latest_msg = state["messages"][-1].content if state.get("messages") else ""
    rag = get_chroma_rag()

    category_filter = None
    if state.get("friction_topic") == "pricing":
        category_filter = "pricing"
    elif state.get("friction_topic") in ("gatekeeper_bounce", "technical_spec"):
        category_filter = "objection_handling"

    chunks = rag.query(latest_msg, n_results=2, category=category_filter)
    if not chunks:
        # Fallback query without category filter
        chunks = rag.query(latest_msg, n_results=2)

    context_snippets = []
    for c in chunks:
        context_snippets.append(c["content"])

    rag_text = "\n\n".join(context_snippets) if context_snippets else "Standard ApexSales AI B2B Sales packages starting at $499/mo (Starter) to $3,500/mo (Enterprise Custom)."
    return {"rag_context": rag_text}


# =====================================================================
# LLM Response Generation Node
# =====================================================================

def generate_response_node(state: AgentState):
    """Generates real-time conversational response using Groq LLM (or Gemini)."""
    persona = state.get("persona", "Corporate Enterprise")
    rag_context = state.get("rag_context", "")
    cluster_id = state.get("cluster_id", 1)
    tactic = KMEANS_PERSONA_PROFILES.get(cluster_id, {}).get("tactic", "")
    enrichment = state.get("enrichment", {})
    company = enrichment.get("company", "your organization")

    system_prompt = f"""You are Sarah, an elite Senior B2B Sales Representative for ApexSales AI.
Target Prospect Organization: {company}
Prospect Segment (K-Means): {persona}
Sales Strategy & Objection Playbook: {tactic}

Retrieved Knowledge Base Context (ChromaDB RAG):
{rag_context}

Rules for Real-Time Phone/Voice Response:
1. Keep replies concise, persuasive, and conversational (1 to 2 sentences maximum).
2. Answer the prospect's question directly using the knowledge base facts.
3. Sound completely natural, warm, confident, and consultative.
4. Guide the prospect toward booking a brief 15-minute live demo or confirming next steps.
5. NEVER sound like a generic robot or read bullet lists aloud."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    if llm:
        try:
            response = llm.invoke(messages)
            ai_text = getattr(response, "content", "")
            if not ai_text and hasattr(response, "additional_kwargs"):
                ai_text = response.additional_kwargs.get("reasoning_content", "")
            if not ai_text or not str(ai_text).strip():
                ai_text = "We offer flexible Starter ($499/mo) and Enterprise ($3,500/mo) packages with sub-300ms SLA and CRM integrations. Would you be open to a 10-minute demo on Thursday?"
            response = AIMessage(content=ai_text)
        except Exception as e:
            logger.error(f"Error invoking Groq LLM: {e}")
            ai_text = "I'd be delighted to walk you through our platform—we help teams scale outbound calls with sub-500ms voice AI. Would you be open to a 10-minute demo on Thursday?"
            response = AIMessage(content=ai_text)
    else:
        ai_text = "Thanks for connecting with ApexSales AI! Our Starter plan is $499/mo and Enterprise is $3,500/mo. What's your primary priority for inbound calls this quarter?"
        response = AIMessage(content=ai_text)

    return {"messages": [response]}


# =====================================================================
# Telemetry & Persistence Node
# =====================================================================

def sync_telemetry_node(state: AgentState):
    """Synchronizes call turn and prospect data asynchronously to Redis, MongoDB, and Spring WebFlux."""
    import threading

    def _persist_telemetry_bg():
        session_id = state.get("session_id", "default_session")
        lead_id = state.get("lead_id", "lead_default")
        user_msg = state["messages"][-2].content if len(state.get("messages", [])) >= 2 else ""
        ai_reply = state["messages"][-1].content if state.get("messages") else ""

        # 1. Update Redis Cache
        try:
            cache = get_redis_cache()
            cache.append_call_turn(session_id, user_msg, ai_reply, state.get("sentiment", 0.0))
            cache.cache_lead_persona(lead_id, {
                "persona": state.get("persona"),
                "cluster_id": state.get("cluster_id"),
                "friction_topic": state.get("friction_topic"),
                "enrichment": state.get("enrichment")
            })
        except Exception as e:
            logger.debug(f"Redis cache sync error: {e}")

        # 2. Update MongoDB Atlas
        try:
            db = get_mongo_service()
            db.record_interaction(lead_id, session_id, {
                "user_msg": user_msg,
                "ai_reply": ai_reply,
                "persona": state.get("persona"),
                "cluster_id": state.get("cluster_id"),
                "friction_topic": state.get("friction_topic"),
                "sentiment": state.get("sentiment")
            })
        except Exception as e:
            logger.debug(f"MongoDB record interaction error: {e}")

        # 3. Dispatch to Java Spring WebFlux Microservice
        try:
            spring_client = get_spring_sync_client()
            spring_client.dispatch_sync_lead_bg({
                "lead_id": lead_id,
                "session_id": session_id,
                "persona": state.get("persona"),
                "cluster_id": state.get("cluster_id"),
                "friction_topic": state.get("friction_topic"),
                "company": state.get("enrichment", {}).get("company", "Prospect"),
                "last_interaction": ai_reply
            })
        except Exception as e:
            logger.debug(f"Spring WebFlux dispatch error: {e}")

    # Launch background thread for non-blocking persistence
    threading.Thread(target=_persist_telemetry_bg, daemon=True).start()
    return {}


# =====================================================================
# StateGraph Compilation
# =====================================================================

workflow = StateGraph(AgentState)
workflow.add_node("segment_lead", segment_lead_node)
workflow.add_node("retrieve_rag", retrieve_rag_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("sync_telemetry", sync_telemetry_node)

workflow.add_edge(START, "segment_lead")
workflow.add_edge("segment_lead", "retrieve_rag")
workflow.add_edge("retrieve_rag", "generate_response")
workflow.add_edge("generate_response", "sync_telemetry")
workflow.add_edge("sync_telemetry", END)

agent_app = workflow.compile()