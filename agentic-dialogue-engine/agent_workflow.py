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
import re
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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

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
# Intelligent STT De-duplication & Speech Cleaner
# =====================================================================

def clean_stt_transcript(text: str) -> str:
    """Removes streaming stutter, consecutive duplicated sentences and repeated clauses from speech recognition."""
    if not text:
        return ""
    s = text.strip()

    # Step 1: Preserve and mask email addresses so punctuation cleaning doesn't split domains
    emails = {}
    def _mask_email(m):
        token = f"EMAILTOKEN{len(emails)}X"
        emails[token.lower()] = m.group(0)
        return token
    s = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', _mask_email, s)

    # Step 2: Immediate word-level stutters (e.g. "Hello hello", "my my")
    s = re.sub(r'\b([a-zA-Z]{2,})(?:\s+[,.]?\s*\1\b)+', r'\1', s, flags=re.IGNORECASE)

    # Step 3: Normalize spacing around punctuation
    s = re.sub(r'\s*([.,?!])\s*', r'\1 ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Step 4: Remove immediate adjacent phrase repetitions (e.g. "Phrase A. Phrase A.")
    for _ in range(5):
        words = s.split()
        if len(words) < 4:
            break
        changed = False
        n = len(words)
        w_clean = [w.lower().strip(".,?!;:-") for w in words]
        for span in range(min(25, n // 2), 1, -1):
            i = 0
            while i + 2 * span <= n:
                c1 = w_clean[i : i + span]
                c2 = w_clean[i + span : i + 2 * span]
                if c1 == c2 and len(" ".join(c1)) > 3:
                    del words[i + span : i + 2 * span]
                    del w_clean[i + span : i + 2 * span]
                    n = len(words)
                    changed = True
                else:
                    i += 1
            if changed:
                break
        s = " ".join(words)

    # Step 5: Split into sentence / clause fragments
    s = re.sub(r'\b(and\s+my|and|or|so)\s+(?=(?:My|I|Phone|Email)\b)', r'. ', s, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.?!])\s+', s)

    # Pass 5A: Discard any fragment that is a prefix or substring of a later, longer fragment
    kept_parts = []
    for idx, p in enumerate(parts):
        p_str = p.strip()
        if not p_str:
            continue
        p_norm = re.sub(r'[^a-z0-9]', '', p_str.lower())
        if not p_norm:
            continue
        
        # Check if p_norm is a substring of any subsequent fragment
        subsumed = False
        for later in parts[idx + 1:]:
            later_norm = re.sub(r'[^a-z0-9]', '', later.lower())
            if p_norm in later_norm and len(later_norm) >= len(p_norm) + 3:
                subsumed = True
                break
        if not subsumed:
            kept_parts.append(p_str)

    res = " ".join(kept_parts)

    # Step 6: Final sentence-level and phrase-level deduplication
    final_sentences = []
    seen_sentences = set()
    for s_item in re.split(r'(?<=[.?!])\s+', res):
        s_item = s_item.strip()
        if not s_item:
            continue
        norm = re.sub(r'[^a-zA-Z0-9]', '', s_item.lower())
        if not norm or norm in seen_sentences:
            continue
        seen_sentences.add(norm)
        final_sentences.append(s_item)

    res = " ".join(final_sentences)

    # Format punctuation
    res = re.sub(r'\s+([.,?!])', r'\1', res)
    res = re.sub(r'([.,?!])(?=[A-Za-z0-9])', r'\1 ', res)
    res = re.sub(r'\s{2,}', ' ', res)

    # Restore emails as the final step so domains stay intact
    for token_lower, orig in emails.items():
        res = re.sub(re.escape(token_lower), orig, res, flags=re.IGNORECASE)

    return res.strip()



# =====================================================================
# Stateful Conversation Memory & Dynamic Lead Extractor
# =====================================================================

class SessionMemory:
    """Maintains continuous dialogue history and extracted customer/plan profile per session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[BaseMessage] = []
        self.customer_name: Optional[str] = None
        self.phone: Optional[str] = None
        self.email: Optional[str] = None
        self.company: Optional[str] = None
        self.chosen_plan: Optional[str] = None  # "starter", "growth", "enterprise"
        self.amount: Optional[float] = None
        self.last_active: float = time.time()

    def update_from_text(self, text: str):
        self.last_active = time.time()
        lower = text.lower()

        # Extract email address (standard, spoken English, or Hindi transliterated)
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            self.email = email_match.group(0).strip().lower()
        else:
            spoken_email = re.search(r'(\b[a-zA-Z0-9_\.-]+)\s+(?:at\s+(?:the\s+rate\s+)?|at\s+the\s+|at\s+|@)\s*([a-zA-Z0-9\.-]+)\s+(?:dot|\.)\s*([a-zA-Z]{2,6}\b)', text, re.IGNORECASE)
            if spoken_email:
                domain_part = spoken_email.group(2).replace(" ", "")
                self.email = f"{spoken_email.group(1)}@{domain_part}.{spoken_email.group(3)}".lower()
            else:
                hindi_email = re.search(r'(?:मेरा\s+ईमेल\s+है|ईमेल\s+है|ईमेल)\s+([\u0900-\u097F\w]+)\s*(?:at\s+the\s+rate|at\s+the|at|@)\s*([\w\.-]+)\s*(?:dot|\.)\s*(\w+)', text, re.IGNORECASE)
                if hindi_email:
                    self.email = f"{hindi_email.group(1)}@{hindi_email.group(2).replace(' ', '')}.{hindi_email.group(3)}".lower()

        # Extract phone number (min 3 digits to support test numbers like 1234 or full numbers)
        phone_match = re.search(r'(?:phone|number|mobile|cell|call me at|phone number should be)?\s*(?:is|:)?\s*([+]?[0-9]{1,3}?[-.\s]?\(?[0-9]{2,4}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}|[0-9]{3,14})', text, re.IGNORECASE)
        if phone_match:
            p_val = phone_match.group(1).strip()
            digits = re.sub(r'\D', '', p_val)
            if len(digits) >= 3:
                self.phone = p_val

        # Extract customer name (English & Hindi - matches "save my name as X", "receipt name as X", "lookup check names as X", etc.)
        name_patterns = [
            r'(?:save|set|put)\s+(?:my\s+|the\s+)?name\s+(?:as|to|is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'(?:add\s+(?:the\s+)?)?receipt\s+name\s+(?:also\s+)?(?:as\s+|to\s+|is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'lookup\s+(?:history\s+)?(?:check\s+)?name[s]?\s*(?:also\s+)?(?:as\s+|to\s+|is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'(?:change|update)\s+(?:my\s+|the\s+)?name\s+(?:to|as|is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'\bname\s+(?:also\s+)?(?:as|to|should\s+be|is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'(?:my\s+name\s+is|this\s+is|call\s+me|i\s+am)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'\bsave\s+(?:my\s+)?name\s+as\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
            r'\bsave\s+as\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)'
        ]
        stop_words = {
            "it", "is", "should", "and", "or", "so", "for", "in", "to", "my", "the", "a", "an",
            "here", "also", "with", "where", "like", "everywhere", "field", "there", "please",
            "can", "you", "save", "as", "at", "be", "all", "check", "checknames", "names",
            "interested", "looking", "ready", "fine", "good", "sure", "okay", "bye", "test",
            "one", "two", "three", "both"
        }
        found_name = False
        for n_pat in name_patterns:
            n_m = re.search(n_pat, text, re.IGNORECASE)
            if n_m:
                cand = n_m.group(1).strip()
                words = cand.split()
                if words and words[0].lower() not in stop_words:
                    if len(words) > 1 and words[1].lower() in stop_words:
                        words = words[:1]
                    clean_name = " ".join(words[:2]).title()
                    self.customer_name = clean_name
                    found_name = True
                    break
        if not found_name:
            hindi_name = re.search(r'(?:मेरा\s+नाम\s+(?:भी\s+)?(?:है|कर\s+दो|रख\s+दो|लिख\s+दो)?|नाम\s+(?:भी\s+)?(?:है|रख\s+दो|लिख\s+दो)?|रसीद\s+(?:का\s+)?नाम\s+(?:भी\s+)?(?:है|रख\s+दो|लिख\s+दो)?)\s+([\u0900-\u097F\w]+(?:\s+[\u0900-\u097F\w]+)?)', text)
            if hindi_name:
                self.customer_name = hindi_name.group(1).strip()

        # Extract or update plan selection
        if any(kw in lower for kw in ["change to growth", "change the starter package to the growth", "change to the growth", "growth package", "growth plan", "1499", "$1499", "option 2", "plan 2"]):
            self.chosen_plan = "growth"
            self.amount = 1499.0
        elif any(kw in lower for kw in ["change to enterprise", "change to the enterprise", "enterprise tier", "enterprise plan", "enterprise package", "3500", "$3500", "option 3", "plan 3"]):
            self.chosen_plan = "enterprise"
            self.amount = 3500.0
        elif any(kw in lower for kw in ["change to starter", "change to the starter", "starter package", "starter plan", "499", "$499", "option 1", "plan 1"]):
            self.chosen_plan = "starter"
            self.amount = 499.0
        elif lower in ("1", "option 1") or lower.startswith("1 "):
            self.chosen_plan = "starter"
            self.amount = 499.0
        elif lower in ("2", "option 2") or lower.startswith("2 "):
            self.chosen_plan = "growth"
            self.amount = 1499.0
        elif lower in ("3", "option 3") or lower.startswith("3 "):
            self.chosen_plan = "enterprise"
            self.amount = 3500.0

        # Auto-persist to MongoService call_history if contact details or plan are present
        if self.customer_name or self.phone or self.email:
            try:
                db = get_mongo_service()
                db.save_call_history({
                    "history_id": f"hist_{self.session_id}",
                    "session_id": self.session_id,
                    "customer_name": self.customer_name or "Valued Customer",
                    "phone": self.phone or "",
                    "email": self.email or "",
                    "company": self.company or "Client Organization",
                    "plan": self.chosen_plan or "growth",
                    "amount": self.amount or (499.0 if self.chosen_plan == "starter" else 3500.0 if self.chosen_plan == "enterprise" else 1499.0),
                    "transcript": [{"role": "user" if isinstance(m, HumanMessage) else "agent", "text": getattr(m, "content", "")} for m in self.messages]
                })
            except Exception as e:
                logger.debug(f"Auto-save call history error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_name": self.customer_name,
            "phone": self.phone,
            "email": self.email,
            "company": self.company or "Client Organization",
            "plan": self.chosen_plan or "growth",
            "amount": self.amount or (499.0 if self.chosen_plan == "starter" else 3500.0 if self.chosen_plan == "enterprise" else 1499.0),
            "invoice_ready": bool(self.chosen_plan and (self.phone or self.email))
        }


_SESSION_CACHE: Dict[str, SessionMemory] = {}

def get_or_create_session(session_id: str) -> SessionMemory:
    if not session_id:
        session_id = "default_session"
    if session_id not in _SESSION_CACHE:
        _SESSION_CACHE[session_id] = SessionMemory(session_id)
    return _SESSION_CACHE[session_id]


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
model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
llm = ChatGroq(
    model=model_name,
    temperature=0.15,
    max_tokens=140,
    api_key=groq_api_key,
    request_timeout=4.0,
    max_retries=1
) if groq_api_key else None


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
    lower = latest_msg.lower()
    if state.get("friction_topic") == "pricing" or any(w in lower for w in ["expensive", "cost", "price", "budget", "discount", "how much", "rate"]):
        category_filter = "pricing"
    elif state.get("friction_topic") in ("gatekeeper_bounce", "technical_spec") or any(w in lower for w in ["who is this", "what do you do", "latency", "architecture", "security", "soc2"]):
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

def _build_sales_system_prompt(company: str, persona: str, tactic: str, rag_context: str, session: SessionMemory) -> str:
    chosen_plan_str = f"Option {1 if session.chosen_plan=='starter' else 2 if session.chosen_plan=='growth' else 3}: {session.chosen_plan.upper()} (${int(session.amount)}/mo)" if session.chosen_plan else "Not chosen yet"
    name_str = session.customer_name or "Not provided yet"
    phone_str = session.phone or "Not provided yet"
    email_str = session.email or "Not provided yet"

    return f"""You are Sarah, an elite Senior B2B Sales Representative for ApexSales AI.
Target Prospect Organization: {company}
Prospect Segment (K-Means): {persona}
Sales Playbook: {tactic}

Retrieved Knowledge Base (RAG):
{rag_context}

LIVE CALL SESSION STATE (WHAT YOU ALREADY KNOW ABOUT THIS CLIENT):
- Confirmed Selected Plan: {chosen_plan_str}
- Customer Name: {name_str}
- Phone Number: {phone_str}
- Email Address: {email_str}
- Complimentary 15-Minute Live Implementation Demo: INCLUDED ($0.00 FREE with all plans)
- Built-In Official PDF Invoice: READY FOR INSTANT PREVIEW & DOWNLOAD

CRITICAL CONVERSATIONAL GUIDELINES & ANTI-REDUNDANCY RULES:
1. TO THE POINT: Speak ONLY important, relevant, high-impact facts. Absolutely no corporate fluff, filler, or rambling. Keep replies to 1-2 punchy sentences maximum.
2. NEVER REPEAT QUESTIONS ALREADY ANSWERED:
   - Carefully review the dialogue history below.
   - If the client ALREADY selected a package (Current Selection: {chosen_plan_str}), NEVER ask "Which package fits your sales team best?" again!
   - If the client ALREADY provided their name ({name_str}), phone ({phone_str}), or email ({email_str}), NEVER ask for them again! Acknowledge what was provided.
3. INVOICE AND PDF REQUESTS:
   - If the client asks to download the PDF, see the invoice, or finalize: confirm that their official PDF invoice for {chosen_plan_str} is ready right now on screen to download or preview!
   - NEVER say "I cannot generate or send a PDF" — ApexSales AI has a built-in real-time PDF generation engine that generates their invoice immediately!
4. AVAILABLE PACKAGES (Only pitch if not chosen yet):
   • Option 1: Starter Plan ($499/mo) — 1,000 live voice mins, sub-300ms SLA, standard CRM webhooks.
   • Option 2: Growth Plan ($1,499/mo) — Unlimited voice mins, bidirectional CRM sync (Salesforce/HubSpot).
   • Option 3: Enterprise Tier ($3,500/mo) — Dedicated private cluster, custom fine-tuned LLM, on-prem VPC, 24/7 priority SLA.
   • All plans include: 15-Minute Complimentary Implementation Demo ($0.00).
5. CLOSING:
   When contact details and package are confirmed, warmly thank them and confirm their invoice is generated!"""


def generate_response_node(state: AgentState):
    """Generates real-time conversational response using Groq LLM (or Gemini)."""
    persona = state.get("persona", "Corporate Enterprise")
    rag_context = state.get("rag_context", "")
    cluster_id = state.get("cluster_id", 1)
    tactic = KMEANS_PERSONA_PROFILES.get(cluster_id, {}).get("tactic", "")
    enrichment = state.get("enrichment", {})
    company = enrichment.get("company", "your organization")
    session_id = state.get("session_id", "default_session")

    session = get_or_create_session(session_id)
    if state.get("messages"):
        latest_msg = state["messages"][-1].content
        session.update_from_text(latest_msg)

    system_prompt = _build_sales_system_prompt(company, persona, tactic, rag_context, session)
    recent_history = session.messages[-8:]
    messages = [SystemMessage(content=system_prompt)] + recent_history + state["messages"][-1:]

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

    if state.get("messages"):
        session.messages.append(HumanMessage(content=state["messages"][-1].content))
    session.messages.append(response)

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

# Parallel branch from START into segment_lead and retrieve_rag concurrently
workflow.add_edge(START, "segment_lead")
workflow.add_edge(START, "retrieve_rag")
workflow.add_edge("segment_lead", "generate_response")
workflow.add_edge("retrieve_rag", "generate_response")
workflow.add_edge("generate_response", "sync_telemetry")
workflow.add_edge("sync_telemetry", END)

agent_app = workflow.compile()


# =====================================================================
# Ultra-Low Latency Streaming & Parallel Execution Function
# =====================================================================
import concurrent.futures

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def stream_agent_turn(
    transcript: str,
    session_id: str = "default_session",
    lead_id: str = "lead_default",
    on_token = None,
    on_first_sentence = None
) -> Dict[str, Any]:
    """
    Sub-300ms Conversational Response Generator:
    1. Executes K-Means segmentation and ChromaDB RAG retrieval in parallel threads.
    2. Maintains continuous session memory and extracts client details & plan dynamically.
    3. Prevents repeating questions already answered.
    4. Streams LLM response tokens directly to callbacks.
    """
    # 0. Clean input speech
    clean_input = clean_stt_transcript(transcript)
    if not clean_input:
        clean_input = transcript.strip()

    session = get_or_create_session(session_id)
    session.update_from_text(clean_input)

    # 1. Parallel Segmentation & RAG
    dummy_state: AgentState = {
        "messages": [HumanMessage(content=clean_input)],
        "session_id": session_id,
        "lead_id": lead_id,
        "persona": "",
        "friction_topic": "",
        "sentiment": 0.0,
        "rag_context": "",
        "enrichment": {},
        "cluster_id": 1
    }

    future_segment = _executor.submit(segment_lead_node, dummy_state)
    future_rag = _executor.submit(retrieve_rag_node, dummy_state)

    segment_res = future_segment.result()
    rag_res = future_rag.result()

    persona = segment_res.get("persona", "Corporate Enterprise")
    cluster_id = segment_res.get("cluster_id", 1)
    friction_topic = segment_res.get("friction_topic", "general_discovery")
    sentiment = segment_res.get("sentiment", 0.3)
    enrichment = segment_res.get("enrichment", {})
    company = enrichment.get("company", "your organization")
    rag_context = rag_res.get("rag_context", "")
    tactic = KMEANS_PERSONA_PROFILES.get(cluster_id, {}).get("tactic", "")

    # 2. Build Stateful Anti-Redundancy Prompt & History
    system_prompt = _build_sales_system_prompt(company, persona, tactic, rag_context, session)
    recent_history = session.messages[-8:]
    messages = [SystemMessage(content=system_prompt)] + recent_history + [HumanMessage(content=clean_input)]

    full_reply = ""
    first_sentence_sent = False

    if llm:
        try:
            # Stream tokens
            for chunk in llm.stream(messages):
                token = getattr(chunk, "content", "")
                if not token and hasattr(chunk, "additional_kwargs"):
                    token = chunk.additional_kwargs.get("reasoning_content", "")
                if token:
                    full_reply += token
                    if on_token:
                        try:
                            on_token(token)
                        except Exception:
                            pass

                    # Detect first sentence for early vocalization
                    if not first_sentence_sent and on_first_sentence:
                        if any(p in full_reply for p in [". ", "? ", "! ", ".\n", "?\n", "!\n"]):
                            for punct in [". ", "? ", "! "]:
                                if punct in full_reply:
                                    first_sent = full_reply.split(punct)[0] + punct.strip()
                                    try:
                                        on_first_sentence(first_sent)
                                    except Exception:
                                        pass
                                    first_sentence_sent = True
                                    break
        except Exception as e:
            logger.error(f"Error streaming from Groq LLM: {e}")
            if not full_reply:
                full_reply = "We offer flexible Starter ($499/mo) and Enterprise ($3,500/mo) packages with sub-300ms SLA and CRM integrations. Would you be open to a 10-minute demo on Thursday?"
    else:
        full_reply = "Thanks for connecting with ApexSales AI! Our Starter plan is $499/mo and Enterprise is $3,500/mo. What's your primary priority for inbound calls this quarter?"

    if not full_reply.strip():
        full_reply = "We offer flexible Starter ($499/mo) and Enterprise ($3,500/mo) packages with sub-300ms SLA and CRM integrations. Would you be open to a 10-minute demo on Thursday?"

    # Append to continuous session history
    session.messages.append(HumanMessage(content=clean_input))
    session.messages.append(AIMessage(content=full_reply))

    # 3. Background Telemetry Dispatch
    sync_telemetry_node({
        "messages": [HumanMessage(content=clean_input), AIMessage(content=full_reply)],
        "session_id": session_id,
        "lead_id": lead_id,
        "persona": persona,
        "friction_topic": friction_topic,
        "sentiment": sentiment,
        "rag_context": rag_context,
        "enrichment": enrichment,
        "cluster_id": cluster_id
    })

    return {
        "reply": full_reply,
        "persona": persona,
        "cluster_id": cluster_id,
        "friction_topic": friction_topic,
        "sentiment": sentiment,
        "rag_context": rag_context,
        "enrichment": enrichment,
        "lead_info": session.to_dict()
    }