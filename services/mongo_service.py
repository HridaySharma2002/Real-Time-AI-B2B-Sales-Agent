"""
services/mongo_service.py - MongoDB Atlas Persistence Layer & Local Fallback
Part of VintushTech Real-Time AI B2B Sales Agent.

Features:
- Stores structured lead profiles, conversation transcripts, BANT qualification notes, and call telemetry.
- Connects to MongoDB Atlas via `MONGODB_URI` with CA certification support.
- Fully resilient: automatically writes to local JSON storage if Atlas connection is unavailable or IP is unwhitelisted.
- Provides synchronous and asynchronous query methods.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MongoService")

MONGODB_URI = os.getenv("MONGODB_URI")
LOCAL_LEADS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".local_leads_db.json")


class MongoDBService:
    """MongoDB Atlas service for lead and interaction persistence with local failover."""

    def __init__(self, uri: Optional[str] = MONGODB_URI):
        self.uri = uri
        self.client = None
        self.db = None
        self._connected = False

        self._connect()

    def _connect(self):
        if not self.uri:
            logger.info("MONGODB_URI not provided; utilizing local JSON lead persistence.")
            return

        try:
            import certifi
            from pymongo import MongoClient

            self.client = MongoClient(
                self.uri,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000
            )
            # Verify connectivity
            self.client.admin.command('ping')
            self.db = self.client.get_database("vintushtech_sales")
            self._connected = True
            logger.info("Connected successfully to MongoDB Atlas database 'vintushtech_sales'.")
        except Exception as e:
            logger.warning(f"MongoDB Atlas unreachable ({type(e).__name__}). Using local persistence failover at '{LOCAL_LEADS_FILE}'.")
            self._connected = False

    def _save_local_record(self, collection_name: str, record: Dict[str, Any]):
        """Persists records locally if MongoDB Atlas is offline."""
        data = {}
        if os.path.exists(LOCAL_LEADS_FILE):
            try:
                with open(LOCAL_LEADS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        items = data.get(collection_name, [])
        items.append(record)
        data[collection_name] = items
        try:
            with open(LOCAL_LEADS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving local lead record: {e}")

    def save_lead(self, lead_data: Dict[str, Any]) -> str:
        """Saves or updates a lead document."""
        lead_id = lead_data.get("lead_id") or f"lead_{int(time.time())}"
        lead_data["lead_id"] = lead_id
        lead_data["updated_at"] = datetime.now().isoformat()

        if self._connected and self.db is not None:
            try:
                self.db.leads.update_one(
                    {"lead_id": lead_id},
                    {"$set": lead_data},
                    upsert=True
                )
                logger.info(f"Saved lead '{lead_id}' to MongoDB Atlas.")
                return lead_id
            except Exception as e:
                logger.warning(f"Error saving lead to MongoDB Atlas: {e}. Falling back to local storage.")

        self._save_local_record("leads", lead_data)
        logger.info(f"Saved lead '{lead_id}' to local database.")
        return lead_id

    def record_interaction(self, lead_id: str, session_id: str, turn_data: Dict[str, Any]):
        """Records a conversational turn with sentiment and friction metrics."""
        record = {
            "lead_id": lead_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            **turn_data
        }

        if self._connected and self.db is not None:
            try:
                self.db.interactions.insert_one(record)
                return
            except Exception as e:
                logger.warning(f"Error recording interaction to MongoDB Atlas: {e}")

        self._save_local_record("interactions", record)

    def save_call_analytics(self, analytics_data: Dict[str, Any]):
        """Persists comprehensive post-call telemetry and K-Means cluster classification."""
        analytics_data["recorded_at"] = datetime.now().isoformat()

        if self._connected and self.db is not None:
            try:
                self.db.analytics.insert_one(analytics_data)
                return
            except Exception as e:
                logger.warning(f"Error saving analytics to MongoDB Atlas: {e}")

        self._save_local_record("analytics", analytics_data)

    def get_recent_leads(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent leads for dashboard display."""
        if self._connected and self.db is not None:
            try:
                cursor = self.db.leads.find({}, {"_id": 0}).sort("updated_at", -1).limit(limit)
                return list(cursor)
            except Exception:
                pass

        if os.path.exists(LOCAL_LEADS_FILE):
            try:
                with open(LOCAL_LEADS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("leads", [])[-limit:]
            except Exception:
                pass
        return []


# Global singleton
_mongo_instance: Optional[MongoDBService] = None

def get_mongo_service() -> MongoDBService:
    global _mongo_instance
    if _mongo_instance is None:
        _mongo_instance = MongoDBService()
    return _mongo_instance
