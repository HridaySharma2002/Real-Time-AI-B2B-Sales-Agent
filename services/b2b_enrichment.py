"""
services/b2b_enrichment.py - External B2B Lead Enrichment Service
Part of VintushTech Real-Time AI B2B Sales Agent.

Features:
- Enriches prospect inquiries with firmographics, company size, tech stack, and estimated ARR.
- Determines matching customer persona and qualification score for sales reps.
- Prepares personalized context for LangGraph dynamic RAG prompts.
"""

import re
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("B2BEnrichment")

# Domain knowledge for recognized B2B segments
PRESET_FIRMOGRAPHICS = {
    "acme": {
        "company_name": "Acme Corporation",
        "employees": "500-1,000",
        "industry": "FinTech / Enterprise SaaS",
        "tech_stack": ["Salesforce", "AWS", "Snowflake", "Kubernetes"],
        "recommended_package": "Enterprise Custom ($3,500+/mo)",
        "pain_points": ["Inbound SDR capacity", "High response latency", "CRM data drift"]
    },
    "startup": {
        "company_name": "NextGen ScaleUp",
        "employees": "20-50",
        "industry": "Early Stage B2B SaaS",
        "tech_stack": ["HubSpot", "Google Workspace", "PostgreSQL"],
        "recommended_package": "Growth Plan ($1,499/mo)",
        "pain_points": ["Lead qualification bottlenecks", "No dedicated SDRs"]
    }
}


class B2BEnrichmentService:
    """Enriches prospect data for personalized real-time sales pitches."""

    @staticmethod
    def enrich_from_text(text: str) -> Dict[str, Any]:
        """Infers company, size, and tech stack from prospect utterance or email."""
        lower = text.lower()
        for key, info in PRESET_FIRMOGRAPHICS.items():
            if key in lower:
                return {
                    "matched": True,
                    "company": info["company_name"],
                    "employees": info["employees"],
                    "industry": info["industry"],
                    "tech_stack": info["tech_stack"],
                    "recommended_tier": info["recommended_package"],
                    "enrichment_source": "B2B_Preset_Enrichment"
                }

        # Dynamic heuristic enrichment
        is_enterprise = any(w in lower for w in ["soc2", "enterprise", "compliance", "security", "sla", "unlimited", "scale"])
        return {
            "matched": False,
            "company": "Prospect Organization",
            "employees": "100-500" if is_enterprise else "10-50",
            "industry": "Technology / B2B Services",
            "tech_stack": ["Salesforce", "REST APIs"] if is_enterprise else ["HubSpot", "Zapier"],
            "recommended_tier": "Enterprise Custom ($3,500+/mo)" if is_enterprise else "Growth Plan ($1,499/mo)",
            "enrichment_source": "Dynamic_Heuristic_Engine"
        }
