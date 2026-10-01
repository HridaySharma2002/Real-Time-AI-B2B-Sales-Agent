"""
server.py - Root FastAPI Server Launcher for ApexSales AI

Usage:
  python server.py
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.join(root_dir, "agentic-dialogue-engine")

for p in [engine_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

main_file = os.path.join(engine_dir, "main.py")
spec = importlib.util.spec_from_file_location("main", main_file)
mod = importlib.util.module_from_spec(spec)
sys.modules["main"] = mod
spec.loader.exec_module(mod)

app = mod.app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  🚀 Starting ApexSales AI Real-Time Voice Server...")
    print("  🌐 Web UI: http://localhost:8000")
    print("  📡 WebSocket: ws://localhost:8000/ws/audio")
    print("=" * 65 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
