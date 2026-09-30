"""
services/mongo_service.py - MongoDB Atlas Persistence Layer & Local Fallback
Part of ApexSales AI Real-Time B2B Sales Agent.

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
import urllib.parse
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MongoService")

LOCAL_LEADS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".local_leads_db.json")


def _resolve_mongo_uri(raw_uri: Optional[str] = None) -> Optional[str]:
    uri = raw_uri or os.getenv("MONGODB_URI")
    user = os.getenv("MONGODB_USERNAME")
    pw = os.getenv("MONGODB_PASSWORD")
    if uri and user and pw:
        enc_u = urllib.parse.quote_plus(user)
        enc_p = urllib.parse.quote_plus(pw)
        if "<db_username>" in uri or "<db_password>" in uri:
            uri = uri.replace("<db_username>", enc_u).replace("<db_password>", enc_p)
    return uri


class MongoDBService:
    """MongoDB Atlas service for lead and interaction persistence with local failover."""

    def __init__(self, uri: Optional[str] = None):
        self.uri = _resolve_mongo_uri(uri)
        self.client = None
        self.db = None
        self._connected = False

        self._connect()

    def is_connected(self) -> bool:
        """Returns True if connected to live MongoDB Atlas, False if running on local failover."""
        return self._connected

    def _connect(self):
        if not self.uri:
            logger.info("MONGODB_URI not configured; utilizing local JSON lead persistence.")
            return

        def _try_connect():
            try:
                import certifi
                from pymongo import MongoClient

                client = MongoClient(
                    self.uri,
                    tlsCAFile=certifi.where(),
                    serverSelectionTimeoutMS=2000,
                    connectTimeoutMS=2000,
                    socketTimeoutMS=2000
                )
                client.admin.command('ping')
                self.client = client
                self.db = client.get_database("apexsales_db")
                self._connected = True
                logger.info("Connected successfully to MongoDB Atlas database 'apexsales_db'.")
            except Exception as e:
                err_str = str(e)
                if "TLSV1_ALERT_INTERNAL_ERROR" in err_str or "SSL" in err_str:
                    logger.warning(
                        "MongoDB Atlas rejected TLS handshake (IP address not whitelisted in Atlas Network Access). "
                        f"Resilient failover active: saving leads locally to '{LOCAL_LEADS_FILE}'."
                    )
                else:
                    logger.warning(f"MongoDB Atlas unreachable ({type(e).__name__}). Using local persistence failover at '{LOCAL_LEADS_FILE}'.")
                self._connected = False

        import threading
        t = threading.Thread(target=_try_connect, daemon=True)
        t.start()
        t.join(timeout=3.0)
        if t.is_alive():
            logger.warning(
                "MongoDB Atlas connection timed out after 3.0s (awaiting IP whitelist in Atlas Network Access). "
                f"Resilient failover active: saving leads locally to '{LOCAL_LEADS_FILE}'."
            )
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

    def get_lead(self, lead_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single lead by lead_id."""
        if self._connected and self.db is not None:
            try:
                doc = self.db.leads.find_one({"lead_id": lead_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception:
                pass

        if os.path.exists(LOCAL_LEADS_FILE):
            try:
                with open(LOCAL_LEADS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for l in reversed(data.get("leads", [])):
                    if l.get("lead_id") == lead_id:
                        return l
            except Exception:
                pass
        return None

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
