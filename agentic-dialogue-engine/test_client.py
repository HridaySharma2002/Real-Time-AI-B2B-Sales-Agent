import asyncio
import json
import wave
import os
import sys
import websockets
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/audio"
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "test_sample_16k.wav")

def check_server_health():
    print(f"[1/3] Checking server health at {BASE_URL}...")
    try:
        req = urllib.request.urlopen(BASE_URL, timeout=3)
        res = json.loads(req.read().decode("utf-8"))
        print(f"      Server response: {res.get('status')} - {res.get('message')}")
        return True
    except Exception as e:
        print(f"      Failed to connect to server: {e}")
        print("      Please make sure the server is running with: uvicorn main:app --reload")
        return False

async def run_websocket_test():
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Test audio file not found at {AUDIO_FILE}")
        return

    print(f"[2/3] Connecting to WebSocket endpoint: {WS_URL}...")
    async with websockets.connect(WS_URL) as ws:
        print("      Connected to WebSocket successfully!")
        
        agent_responded = asyncio.Event()

        async def receive_messages():
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    if msg_type == "transcript":
                        text = data.get("text", "")
                        is_final = data.get("is_final", False)
                        status = "[FINAL]" if is_final else "[PARTIAL]"
                        print(f"      [STT {status}]: {text}")
                    elif msg_type == "agent_response":
                        response_text = data.get("text", "")
                        print(f"\n      >>> [AI SALES AGENT RESPONSE]:\n      \"{response_text}\"\n")
                        agent_responded.set()
            except websockets.ConnectionClosed:
                pass
            except Exception as e:
                print(f"      Receive error: {e}")

        recv_task = asyncio.create_task(receive_messages())

        print(f"[3/3] Streaming audio from '{os.path.basename(AUDIO_FILE)}'...")
        with wave.open(AUDIO_FILE, "rb") as w:
            chunk_size = 1600 # 100ms chunks at 16kHz 16-bit
            while True:
                data = w.readframes(chunk_size)
                if not data:
                    break
                await ws.send(data)
                await asyncio.sleep(0.08)

        # Send silence to signal end of turn
        await ws.send(b"\x00" * 32000)

        print("      Waiting for AI agent response...")
        try:
            await asyncio.wait_for(agent_responded.wait(), timeout=10.0)
            print("      Test completed successfully! Both STT and Agent Response verified.")
        except asyncio.TimeoutError:
            print("      Timed out waiting for agent response.")
        finally:
            recv_task.cancel()

def main():
    if not check_server_health():
        sys.exit(1)
    asyncio.run(run_websocket_test())

if __name__ == "__main__":
    main()
