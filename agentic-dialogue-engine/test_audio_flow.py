import os
import time
import wave
from dotenv import load_dotenv

load_dotenv()

from assemblyai.streaming.v3 import (
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    RealTimeParameters,
    RealTimeEvents,
)
from agent_workflow import agent_app
from langchain_core.messages import HumanMessage

api_key = os.getenv("ASSEMBLYAI_API_KEY")
print(f"Loaded AssemblyAI API Key: {api_key[:6]}...")

transcriber = RealTimeTranscriber(RealTimeTranscriberOptions(api_key=api_key))

def on_turn(client, event):
    print(f"[AssemblyAI Turn] transcript='{event.transcript}', end_of_turn={event.end_of_turn}")
    if event.end_of_turn and event.transcript.strip():
        print(f"[User Said]: {event.transcript}")
        result = agent_app.invoke({"messages": [HumanMessage(content=event.transcript)]})
        agent_response = result["messages"][-1].content
        print(f"[Agent Response]: {agent_response}")

def on_error(client, error):
    print(f"[AssemblyAI Error]: {error}")

transcriber.on(RealTimeEvents.Turn, on_turn)
transcriber.on(RealTimeEvents.Error, on_error)

print("Connecting to AssemblyAI...")
transcriber.connect(RealTimeParameters(sample_rate=16000))
print("Connected! Streaming audio...")

with wave.open("test_sample_16k.wav", "rb") as w:
    chunk_size = 1600  # 100ms at 16kHz 16-bit mono = 3200 bytes
    while True:
        data = w.readframes(chunk_size)
        if not data:
            break
        transcriber.stream(data)
        time.sleep(0.08)

# Send 1 second of silence for turn completion
transcriber.stream(b"\x00" * 32000)
time.sleep(3.0)

print("Disconnecting...")
transcriber.disconnect(terminate=True)
print("Finished test successfully!")
