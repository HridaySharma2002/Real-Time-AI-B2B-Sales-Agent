"""
agentic-dialogue-engine/stt_service.py - AssemblyAI Real-Time STT + TTS Audio Streaming
Part of ApexSales AI Real-Time B2B Sales Agent.

Features:
- Connects to AssemblyAI Streaming v3 WebSocket for real-time speech-to-text.
- Uses dynamic turn detection (min_turn_silence=500ms, max_turn_silence=1000ms).
- Auto-reconnects on network drops or inactivity timeouts.
- Triggers LangGraph multi-step sales agent upon end-of-turn.
- Synthesizes sales agent voice using Chatterbox / Kokoro / Edge-TTS.
- Delivers transcripts, rich metadata (persona, friction, RAG chunks), and base64 audio to WebSocket clients.
"""

import os
import sys
import base64
import logging
import threading
from dotenv import load_dotenv

# Ensure root directory is accessible for services and TTS
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load root .env first, then local if present
load_dotenv(os.path.join(root_dir, ".env"))
load_dotenv()

from assemblyai.streaming.v3 import (
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    RealTimeParameters,
    RealTimeEvents,
)

logger = logging.getLogger("STTService")


_cached_tts_agent = None

def _get_tts_agent():
    """Dynamically loads and caches Chatterbox / Edge-TTS agent."""
    global _cached_tts_agent
    if _cached_tts_agent is not None:
        return _cached_tts_agent

    try:
        import importlib.util
        tts_file = os.path.join(root_dir, "chatterbox-tts.py")
        if os.path.exists(tts_file):
            spec = importlib.util.spec_from_file_location("chatterbox_tts", tts_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _cached_tts_agent = mod.ChatterboxTTSAgent(voice_preset="consultative_rep_female")
            return _cached_tts_agent
    except Exception as e:
        logger.warning(f"Error loading ChatterboxTTS: {e}")
    return None


class TranscriberService:
    def __init__(self, on_transcript=None, on_agent_response=None, sample_rate=16000, session_id=None, lead_id=None):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY", "placeholder_key")
        self.sample_rate = sample_rate
        self.session_id = session_id or "session_default"
        self.lead_id = lead_id or "lead_default"
        self.on_transcript = on_transcript
        self.on_agent_response = on_agent_response
        self.tts = _get_tts_agent()
        self._connected = False
        self._pending_buffer = bytearray()
        self._init_client()

    def _init_client(self):
        self.client = RealTimeTranscriber(RealTimeTranscriberOptions(api_key=self.api_key))
        self.client.on(RealTimeEvents.Turn, self._on_turn)
        self.client.on(RealTimeEvents.Error, self._on_error)
        self.client.on(RealTimeEvents.Termination, self._on_termination)

    def _on_termination(self, client, event):
        logger.info(f"AssemblyAI session terminated: {event}")
        self._connected = False

    def _on_error(self, client, error):
        logger.warning(f"AssemblyAI STT error: {error}")
        self._connected = False

    def _on_turn(self, client, event):
        if not event.transcript:
            return
        
        if self.on_transcript:
            try:
                self.on_transcript(event.transcript, event.end_of_turn)
            except Exception as e:
                logger.error(f"Error in on_transcript callback: {e}")

        if event.end_of_turn and event.transcript.strip():
            # Process turn in a background daemon thread so AssemblyAI's client WebSocket loop is never blocked
            threading.Thread(
                target=self._process_turn_response,
                args=(event.transcript,),
                daemon=True
            ).start()

    def _process_turn_response(self, transcript: str):
        safe_user_transcript = transcript.encode("ascii", "replace").decode("ascii")
        print(f"[User Said]: {safe_user_transcript}", flush=True)
        try:
            from agent_workflow import stream_agent_turn

            def handle_token(token: str):
                if self.on_agent_response:
                    self.on_agent_response({
                        "type": "agent_stream_chunk",
                        "token": token
                    })

            def handle_first_sentence(sentence: str):
                if self.on_agent_response:
                    self.on_agent_response({
                        "type": "agent_first_sentence",
                        "sentence": sentence
                    })

            # Run ultra-fast parallel reasoning + streaming generator
            result = stream_agent_turn(
                transcript=transcript,
                session_id=self.session_id,
                lead_id=self.lead_id,
                on_token=handle_token,
                on_first_sentence=handle_first_sentence
            )

            agent_response = result.get("reply", "")
            if not agent_response or not str(agent_response).strip():
                agent_response = "We offer flexible Starter ($499/mo) and Enterprise ($3,500/mo) packages with sub-300ms SLA and CRM integrations. Would you be open to a 10-minute demo on Thursday?"

            persona = result.get("persona", "Corporate Enterprise")
            cluster_id = result.get("cluster_id", 1)
            friction_topic = result.get("friction_topic", "general_discovery")
            rag_context = result.get("rag_context", "")
            enrichment = result.get("enrichment", {})

            safe_agent_response = agent_response.encode("ascii", "replace").decode("ascii")
            print(f"[Agent Response]: {safe_agent_response}", flush=True)

            # 1. Deliver final structured response to prospect
            if self.on_agent_response:
                self.on_agent_response({
                    "type": "agent_response",
                    "text": agent_response,
                    "user_transcript": transcript,
                    "audio_base64": None,
                    "persona": persona,
                    "cluster_id": cluster_id,
                    "friction_topic": friction_topic,
                    "rag_context": rag_context,
                    "enrichment": enrichment
                })

            # 2. Synthesize server-side neural TTS in background without delaying text dialogue
            if self.tts:
                try:
                    wav_bytes = self.tts.synthesize_to_wav_bytes(agent_response)
                    if wav_bytes and len(wav_bytes) > 200:
                        audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
                        if self.on_agent_response:
                            self.on_agent_response({
                                "type": "agent_audio",
                                "text": agent_response,
                                "audio_base64": audio_base64
                            })
                except Exception as tts_err:
                    logger.debug(f"Background TTS synthesis error: {tts_err}")

        except Exception as e:
            import traceback
            print(f"Error generating agent response: {e}", flush=True)
            traceback.print_exc()
            # Failsafe: Send immediate fallback reply so UI is never left in "Thinking..."
            if self.on_agent_response:
                self.on_agent_response({
                    "type": "agent_response",
                    "text": "Thanks for sharing that! Our Starter plan starts at $499/mo and Enterprise is $3,500/mo. Would you be open to a brief 10-minute demo this Thursday?",
                    "user_transcript": transcript,
                    "audio_base64": None
                })

    def connect(self, sample_rate=None, max_retries=3):
        import time
        rate = sample_rate or self.sample_rate
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self._init_client()
                # Configure turn silence thresholds for low-latency voice detection
                params = RealTimeParameters(
                    sample_rate=rate,
                    min_turn_silence=500,
                    max_turn_silence=1000,
                    end_of_turn_confidence_threshold=0.4
                )
                self.client.connect(params)
                self._connected = True
                logger.info(f"AssemblyAI connected successfully with turn parameters (attempt {attempt})")
                if self._pending_buffer:
                    logger.info(f"Flushing {len(self._pending_buffer)} bytes of queued audio to AssemblyAI")
                    self.client.stream(bytes(self._pending_buffer))
                    self._pending_buffer.clear()
                return
            except Exception as e:
                last_err = e
                logger.warning(f"AssemblyAI connect attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(0.8)
        raise last_err

    def stream(self, data: bytes):
        if not self._connected:
            try:
                self.connect()
            except Exception as e:
                logger.warning(f"Auto-connect on stream failed: {e}")
                if len(self._pending_buffer) < 320000:
                    self._pending_buffer.extend(data)
                return

        try:
            self.client.stream(data)
        except Exception as e:
            logger.warning(f"Error streaming to AssemblyAI: {e}. Reconnecting...")
            self._connected = False
            try:
                self.connect()
                self.client.stream(data)
            except Exception as retry_err:
                logger.error(f"Reconnection stream failed: {retry_err}")
                if len(self._pending_buffer) < 320000:
                    self._pending_buffer.extend(data)

    def disconnect(self, terminate: bool = True):
        self._pending_buffer.clear()
        if self._connected:
            try:
                self.client.disconnect(terminate=terminate)
            except Exception as e:
                logger.warning(f"Error disconnecting transcriber: {e}")
            finally:
                self._connected = False

    def close(self):
        self.disconnect(terminate=True)


def get_transcriber(on_transcript=None, on_agent_response=None, sample_rate=16000, session_id=None, lead_id=None):
    return TranscriberService(
        on_transcript=on_transcript,
        on_agent_response=on_agent_response,
        sample_rate=sample_rate,
        session_id=session_id,
        lead_id=lead_id
    )