import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from stt_service import get_transcriber

app = FastAPI(title="Real-Time AI B2B Sales Agent")

# Enable CORS for external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "running",
        "message": "Real-Time AI B2B Sales Agent is active",
        "endpoints": {
            "root": "/",
            "interactive_test_ui": "/test",
            "sample_audio": "/sample-audio",
            "audio_websocket": "/ws/audio"
        }
    }

@app.get("/sample-audio")
def get_sample_audio():
    sample_file = os.path.join(os.path.dirname(__file__), "test_sample_16k.wav")
    if os.path.exists(sample_file):
        return FileResponse(sample_file, media_type="audio/wav")
    return {"error": "Sample audio file not found"}

@app.get("/test", response_class=HTMLResponse)
def get_test_page():
    # Return an interactive web page to test the mic and sales agent in real-time
    html_file = os.path.join(os.path.dirname(__file__), "test_ui.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Test UI file not found</h1>"

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    def send_transcript(text: str, is_final: bool):
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "is_final": is_final
                }),
                loop
            )
        except Exception as e:
            print(f"Error sending transcript: {e}", flush=True)

    def send_agent_response(response_text: str, user_transcript: str):
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "agent_response",
                    "text": response_text,
                    "user_transcript": user_transcript
                }),
                loop
            )
        except Exception as e:
            print(f"Error sending agent response: {e}", flush=True)

    transcriber = get_transcriber(
        on_transcript=send_transcript,
        on_agent_response=send_agent_response,
        sample_rate=16000
    )
    transcriber.connect()
    print("Transcriber connected for WebSocket client", flush=True)

    try:
        while True:
            data = await websocket.receive_bytes()
            transcriber.stream(data)
    except WebSocketDisconnect:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"WebSocket session error: {e}", flush=True)
    finally:
        transcriber.close()
        print("Transcriber closed", flush=True)