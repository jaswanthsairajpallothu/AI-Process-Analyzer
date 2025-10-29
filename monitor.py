# monitor.py
import time
import psutil
import threading
from collections import defaultdict, deque
import traceback

SAMPLE_INTERVAL = 1.0  # seconds
WINDOW_SIZE = 60       # number of samples to keep per process

class ProcessStore:
    """
    Keeps sliding window of recent samples per pid.
    Each sample is a dict with timestamp and metrics.
    """
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.store = defaultdict(lambda: deque(maxlen=self.window_size))
        self.lock = threading.Lock()

    def add_sample(self, pid, sample):
        with self.lock:
            self.store[pid].append(sample)

    def get_snapshot(self):
        """
        Return a copy of the store (for safe consumption).
        """
        with self.lock:
            return {pid: list(deque_obj) for pid, deque_obj in self.store.items()}

    def cleanup_dead(self):
        # Optionally remove PIDs with empty windows or dead processes
        with self.lock:
            alive_pids = {p.pid for p in psutil.process_iter(attrs=[])}
            for pid in list(self.store.keys()):
                if pid not in alive_pids and len(self.store[pid]) == 0:
                    del self.store[pid]

def sample_once(store: ProcessStore):
    now = time.time()
    for proc in psutil.process_iter(attrs=['pid', 'name', 'username', 'cmdline']):
        try:
            pid = proc.info['pid']
            cpu_percent = proc.cpu_percent(interval=None)  # non-blocking
            mem_info = proc.memory_info()
            mem_percent = proc.memory_percent()
            io_counters = proc.io_counters() if proc.is_running() else None
            num_threads = proc.num_threads()
            open_files = len(proc.open_files()) if proc.is_running() else 0
            sample = {
                'timestamp': now,
                'pid': pid,
                'name': proc.info.get('name'),
                'username': proc.info.get('username'),
                'cmdline': " ".join(proc.info.get('cmdline') or []),
                'cpu_percent': cpu_percent,
                'memory_rss': getattr(mem_info, 'rss', 0),
                'memory_percent': mem_percent,
                'read_bytes': getattr(io_counters, 'read_bytes', 0) if io_counters else 0,
                'write_bytes': getattr(io_counters, 'write_bytes', 0) if io_counters else 0,
                'threads': num_threads,
                'open_files': open_files
            }
            store.add_sample(pid, sample)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            traceback.print_exc()
            continue

def start_sampling(store: ProcessStore, interval=SAMPLE_INTERVAL):
    def run():
        while True:
            sample_once(store)
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t
