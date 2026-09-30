"""
agentic-dialogue-engine/stt_service.py - AssemblyAI Real-Time STT + TTS Audio Streaming
Part of ApexSales AI Real-Time B2B Sales Agent.

Features:
- Connects to AssemblyAI Streaming v3 WebSocket for real-time speech-to-text.
- Triggers LangGraph multi-step sales agent upon end-of-turn.
- Synthesizes sales agent voice using Chatterbox / Kokoro TTS.
- Delivers transcripts, rich metadata (persona, friction, RAG chunks), and base64 audio to WebSocket clients.
"""

import os
import sys
import base64
import logging
from dotenv import load_dotenv

# Ensure root directory is accessible for services and TTS
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from assemblyai.streaming.v3 import (
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    RealTimeParameters,
    RealTimeEvents,
)

load_dotenv()
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
        self.client = RealTimeTranscriber(RealTimeTranscriberOptions(api_key=self.api_key))
        
        self.client.on(RealTimeEvents.Turn, self._on_turn)
        self.client.on(RealTimeEvents.Error, self._on_error)
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
            safe_user_transcript = event.transcript.encode("ascii", "replace").decode("ascii")
            print(f"[User Said]: {safe_user_transcript}", flush=True)
            try:
                from agent_workflow import agent_app
                from langchain_core.messages import HumanMessage
                
                # Invoke LangGraph Multi-Step Reasoning Agent
                result = agent_app.invoke({
                    "messages": [HumanMessage(content=event.transcript)],
                    "session_id": self.session_id,
                    "lead_id": self.lead_id
                })
                
                agent_response = result["messages"][-1].content
                persona = result.get("persona", "Corporate Enterprise")
                cluster_id = result.get("cluster_id", 1)
                friction_topic = result.get("friction_topic", "general_discovery")
                rag_context = result.get("rag_context", "")
                enrichment = result.get("enrichment", {})

                # Synthesize TTS audio response
                audio_base64 = None
                if self.tts:
                    try:
                        wav_bytes = self.tts.synthesize_to_wav_bytes(agent_response)
                        audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
                    except Exception as tts_err:
                        logger.warning(f"TTS synthesis error: {tts_err}")

                safe_agent_response = agent_response.encode("ascii", "replace").decode("ascii")
                print(f"[Agent Response]: {safe_agent_response}", flush=True)

                if self.on_agent_response:
                    self.on_agent_response({
                        "text": agent_response,
                        "user_transcript": event.transcript,
                        "audio_base64": audio_base64,
                        "persona": persona,
                        "cluster_id": cluster_id,
                        "friction_topic": friction_topic,
                        "rag_context": rag_context,
                        "enrichment": enrichment
                    })

            except Exception as e:
                import traceback
                print(f"Error generating agent response: {e}", flush=True)
                traceback.print_exc()

    def _on_error(self, client, error):
        print(f"STT Error: {error}", flush=True)

    def connect(self, sample_rate=None):
        rate = sample_rate or self.sample_rate
        self.client.connect(RealTimeParameters(sample_rate=rate))
        self._connected = True

    def stream(self, data: bytes):
        if self._connected:
            self.client.stream(data)

    def disconnect(self, terminate: bool = True):
        if self._connected:
            try:
                self.client.disconnect(terminate=terminate)
            except Exception as e:
                print(f"Error disconnecting transcriber: {e}")
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