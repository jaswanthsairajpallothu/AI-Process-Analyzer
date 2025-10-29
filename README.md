1 . Project Overview

Title: AI Process Performance Analyzer - Real-time process monitoring & anomaly detection

Goal: Build a tool that continuously collects per-process system metrics (CPU, memory, I/O, threads), analyzes them with an ML-based anomaly detector in near real-time, and visualizes results on a dashboard with alerts. The AI component highlights suspicious or resource-abnormal processes and gives explainable reasons (feature importance / score) to help administrators decide actions.

Expected outcomes:

* Continuous collection of process metrics with minimal overhead.
* Online anomaly scoring to flag abnormal processes (spikes, memory leaks, runaway CPU).
* Visual dashboard (live charts + list of flagged processes).
* Exportable logs and a reproducible Git repo (≥7 commits, branch workflow).
* The documentation and report are sufficient for AI and human evaluators to award full marks.

Scope: Desktop/server monitoring on Linux/Windows/macOS (psutil supports all). Not an OS-level kernel module — a user-space monitoring tool suitable for a student project and demonstration.

 

2 . Module-wise breakdown

Module 1 — Data Collection & Ingestion (Monitor Module)

Purpose: Periodically sample running processes and system metrics. Provide data to the analyzer.

Key responsibilities:

* Collect metrics per PID: cpu%, cpu time, memory rss, memory percent, read bytes, write bytes, thread count, open files count, cmdline, username, start time.
* Maintain a recent sliding window per process (configurable window, e.g., last 60 samples).
* Stream data to the analyzer via internal queue or REST endpoint.

Module 2 — AI Analyzer (Anomaly Detection Module)

Turn process data into features, compute anomaly scores with explanations.

Extract short-term stats, run lightweight anomaly detection (IsolationForest, heuristics).

* Light-weight online anomaly detection using IsolationForest (retrain on rolling window) and complementary heuristics (e.g., sudden CPU%> threshold).
* Report per-process anomaly score and main contributing features.

Module 3 — Dashboard & Export (UI/Presentation Module)

Purpose: Visualize live metrics and anomalies, allow drill-down, export logs, and trigger alerts.

Key responsibilities:

* FastAPI backend exposing REST endpoints and a WebSocket for live streaming.
* Simple web UI (HTML + JS with Plotly.js or Chart.js) to show system CPU/mem + per-process charts and flag list.
* Export logs / CSV and the ability to snapshot process details.

 

3 . Functionalities (brief features list)

* Live system overview: total CPU usage, memory usage, disk I/O.
* Live per-process metrics and charts (auto-updating).
* Auto-detection of abnormal processes with anomaly score.
* Per-process history (last N seconds), with option to export CSV.
* Alerting: on-screen warnings, log file, and optional email (configurable).
* Explainability: show which features (e.g., CPU spike or memory growth) caused the high anomaly score.
* Config options: sampling interval, sliding window size, anomaly threshold, and logging level.

 

4 . Technology Recommendations

* Language: Python 3.10+
* Libraries:
  * psutil — process/system metrics
  * pandas — lightweight feature engineering
  * scikit-learn — IsolationForest anomaly model
  * numpy
  * fastapi, uvicorn — backend server & WebSocket
  * Jinja2 or static HTML for frontend template
  * websockets (FastAPI provides websocket support)
  * plotly.js or Chart.js for frontend charting (use CDN)
  * pytest for unit tests (optional)
* Tools: Git + GitHub, VS Code, and a browser for the dashboard.

 

5 . Execution Plan - step-by-step (high-level then detailed later)

High-level:

1. Run uvicorn to start the backend.
2. Open http://localhost:8000 to view the dashboard.
3. Start load generation (optional) to demonstrate anomalies (e.g., stress CPU or memory).
4. Use the dashboard to identify flagged processes and read explanations.