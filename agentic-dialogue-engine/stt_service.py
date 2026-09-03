import os
from dotenv import load_dotenv
from assemblyai.streaming.v3 import (
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    RealTimeParameters,
    RealTimeEvents,
)

load_dotenv()

class TranscriberService:
    def __init__(self, on_transcript=None, on_agent_response=None, sample_rate=16000):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.sample_rate = sample_rate
        self.on_transcript = on_transcript
        self.on_agent_response = on_agent_response
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
                print(f"Error in on_transcript callback: {e}")

        if event.end_of_turn and event.transcript.strip():
            safe_user_transcript = event.transcript.encode("ascii", "replace").decode("ascii")
            print(f"[User Said]: {safe_user_transcript}", flush=True)
            try:
                from agent_workflow import agent_app
                from langchain_core.messages import HumanMessage
                
                result = agent_app.invoke({"messages": [HumanMessage(content=event.transcript)]})
                agent_response = result["messages"][-1].content
                
                if self.on_agent_response:
                    self.on_agent_response(agent_response, event.transcript)
                
                safe_agent_response = agent_response.encode("ascii", "replace").decode("ascii")
                print(f"[Agent Response]: {safe_agent_response}", flush=True)
                # Next step: Send agent_response to Tushar's TTS
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

def get_transcriber(on_transcript=None, on_agent_response=None, sample_rate=16000):
    return TranscriberService(
        on_transcript=on_transcript,
        on_agent_response=on_agent_response,
        sample_rate=sample_rate
    )