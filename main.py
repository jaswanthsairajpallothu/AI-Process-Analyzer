# main.py
import uvicorn
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from monitor import ProcessStore, start_sampling
from analyzer import Analyzer

app = FastAPI(title="AI Process Performance Analyzer")
store = ProcessStore(window_size=60)
start_sampling(store, interval=1.0)
analyzer = Analyzer(retrain_interval=10)  # faster retrain for demo

# Simple in-memory subscriber list for websocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        for d in disconnected:
            self.disconnect(d)

manager = ConnectionManager()

@app.get("/")
async def root():
    html_content = open("templates/index.html").read()
    return HTMLResponse(html_content)

@app.get("/api/system_snapshot")
async def get_system_snapshot():
    snap = store.get_snapshot()
    # return summary counts
    return JSONResponse({"num_tracked": len(snap)})

@app.get("/api/analyze")
async def analyze_once(top_n: int = 10):
    snap = store.get_snapshot()
    results = analyzer.analyze_snapshot(snap, top_n=top_n)
    return JSONResponse({"results": results})

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # compute latest analysis and broadcast
            snap = store.get_snapshot()
            results = analyzer.analyze_snapshot(snap, top_n=20)
            out = {
                "timestamp": time.time(),
                "summary": {
                    "num_procs": len(snap)
                },
                "anomalies": results
            }
            await manager.broadcast(out)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
