# AI-Powered Process Performance Analyzer

This project is a real-time web dashboard that monitors all running OS processes and uses an AI model (Isolation Forest) to detect and flag anomalous behavior.

It was built for the CSE 234: Operating Systems course at Lovely Professional University.

## Features

* **Live Process Monitoring:** Tracks CPU, memory, and I/O for all running processes.
* **AI Anomaly Detection:** Uses an `IsolationForest` model to learn "normal" system behavior and flag processes that deviate from it.
* **Real-time Dashboard:** A `FastAPI` backend streams data via `WebSockets` to a live `Plotly.js` dashboard.
* **Explainable AI:** Provides a statistical "reason" (Z-score) for why a process was flagged.

## Technology Stack

* **Backend:** Python 3, FastAPI, Uvicorn
* **Data Streaming:** WebSockets
* **OS Monitoring:** `psutil`
* **AI / Data:** `scikit-learn`, `pandas`, `numpy`
* **Frontend:** HTML, JavaScript, `Plotly.js`

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jaswanthsairajpallothu/AI-Process-Analyzer.git](https://github.com/jaswanthsairajpallothu/AI-Process-Analyzer.git)
    cd AI-Process-Analyzer
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

5.  **Open the dashboard:**
    Open your web browser and go to `http://127.0.0.1:8000`

6.  **(Optional) Generate Test Load:**
    To test the anomaly detection, run the included load generator script in a separate terminal:
    ```bash
    python utils/generate_load.py
    ```
