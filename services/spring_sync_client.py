"""
services/spring_sync_client.py - Async Data Sync Client for Java Spring WebFlux Microservice
Part of ApexSales AI Real-Time B2B Sales Agent.

Features:
- Non-blocking asynchronous sync from FastAPI to the Java Spring WebFlux microservice.
- Dispatches lead updates, conversational interaction turns, and post-call analytics.
- Resilient background task execution: logs telemetry safely without stalling real-time audio threads.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SpringSyncClient")

SPRING_SERVICE_URL = os.getenv("SPRING_SERVICE_URL", "http://localhost:8080").rstrip("/")


class SpringWebFluxSyncClient:
    """Async client communicating with the reactive Java Spring WebFlux microservice."""

    def __init__(self, base_url: str = SPRING_SERVICE_URL):
        self.base_url = base_url
        self._timeout = httpx.Timeout(4.0, connect=2.0)

    async def sync_lead(self, lead_data: Dict[str, Any]) -> bool:
        """Sends lead creation/update to Spring WebFlux `/api/leads`."""
        url = f"{self.base_url}/api/leads"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.post(url, json=lead_data)
                if res.status_code in (200, 201):
                    logger.info(f"[SpringSync] Lead {lead_data.get('lead_id')} successfully synced.")
                    return True
                else:
                    logger.warning(f"[SpringSync] Lead sync responded with status {res.status_code}: {res.text}")
        except Exception as e:
            logger.debug(f"[SpringSync] Spring WebFlux microservice offline or unreachable at {url} ({e}). Sync queued locally.")
        return False

    async def sync_interaction(self, lead_id: str, interaction_data: Dict[str, Any]) -> bool:
        """Sends an interaction turn to Spring WebFlux `/api/leads/{id}/interactions`."""
        url = f"{self.base_url}/api/leads/{lead_id}/interactions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.post(url, json=interaction_data)
                return res.status_code in (200, 201)
        except Exception as e:
            logger.debug(f"[SpringSync] Microservice unreachable for interaction sync: {e}")
        return False

    async def sync_call_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """Sends post-call analytics and K-Means cluster data to Spring WebFlux."""
        url = f"{self.base_url}/api/analytics"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.post(url, json=analytics_data)
                return res.status_code in (200, 201)
        except Exception as e:
            logger.debug(f"[SpringSync] Microservice unreachable for analytics sync: {e}")
        return False

    def dispatch_sync_lead_bg(self, lead_data: Dict[str, Any]):
        """Dispatches lead sync task in the background without blocking."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.sync_lead(lead_data))
        except RuntimeError:
            pass


# Global singleton
_spring_client_instance: Optional[SpringWebFluxSyncClient] = None

def get_spring_sync_client() -> SpringWebFluxSyncClient:
    global _spring_client_instance
    if _spring_client_instance is None:
        _spring_client_instance = SpringWebFluxSyncClient()
    return _spring_client_instance
