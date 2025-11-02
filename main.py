# main.py
import uvicorn
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from monitor import ProcessStore, start_sampling # Data collection/storage
from analyzer import Analyzer # Anomaly detection logic

# --- Application Setup and Initialization ---

app = FastAPI(title="AI Process Performance Analyzer")
# Initialize data storage (ProcessStore) and start background metric collection
store = ProcessStore(window_size=60)
start_sampling(store, interval=1.0) # Start the monitoring thread
# Initialize the analyzer, which starts its own background retraining thread
analyzer = Analyzer(retrain_interval=10) # faster retrain for demo

# --- WebSocket Connection Management ---

class ConnectionManager:
    """Handles active WebSocket connections and broadcasting messages."""
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        """Accepts a new connection and adds it to the active list."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Removes a connection from the active list."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Sends a JSON message to all currently connected clients."""
        disconnected = []
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                # Mark disconnected clients for cleanup
                disconnected.append(conn) 
        for d in disconnected:
            self.disconnect(d)

manager = ConnectionManager()

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    """Serves the main HTML page for the application."""
    html_content = open("templates/index.html").read()
    return HTMLResponse(html_content)

@app.get("/api/system_snapshot")
async def get_system_snapshot():
    """Provides a basic summary of the processes currently being tracked."""
    snap = store.get_snapshot()
    return JSONResponse({"num_tracked": len(snap)})

@app.get("/api/analyze")
async def analyze_once(top_n: int = 10):
    """Triggers a one-time analysis and returns the top anomalous processes."""
    snap = store.get_snapshot()
    results = analyzer.analyze_snapshot(snap, top_n=top_n)
    return JSONResponse({"results": results})

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Manages the real-time analysis stream via WebSocket."""
    await manager.connect(websocket)
    try:
        while True:
            # Get the latest process metrics
            snap = store.get_snapshot()
            # Run the anomaly detection
            results = analyzer.analyze_snapshot(snap, top_n=20)
            
            out = {
                "timestamp": time.time(),
                "summary": {
                    "num_procs": len(snap)
                },
                "anomalies": results
            }
            # Broadcast results to all connected clients
            await manager.broadcast(out) 
            # Wait for 1 second before generating the next analysis
            await asyncio.sleep(1.0) 
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        # Catch unexpected errors and ensure disconnect is called
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Standard entry point to run the application using Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
