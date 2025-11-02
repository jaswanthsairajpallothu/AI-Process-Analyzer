# monitor.py
import time
import psutil 
import threading
from collections import defaultdict, deque
import traceback

SAMPLE_INTERVAL = 1.0  # seconds between samples
WINDOW_SIZE = 60       # number of samples to keep per process

class ProcessStore:
    """
    Manages a sliding window of recent performance samples for each PID.
    Uses deque for efficient window management.
    """
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        # Store: {pid: deque of samples}
        self.store = defaultdict(lambda: deque(maxlen=self.window_size))
        self.lock = threading.Lock() # Lock for thread-safe access

    def add_sample(self, pid, sample):
        """Adds a new sample, oldest sample is automatically dropped by deque."""
        with self.lock:
            self.store[pid].append(sample)

    def get_snapshot(self):
        """Returns a thread-safe copy of the store for external use."""
        with self.lock:
            # Convert deques to lists before returning
            return {pid: list(deque_obj) for pid, deque_obj in self.store.items()}

    def cleanup_dead(self):
        """Removes entries for dead processes with empty sample windows."""
        with self.lock:
            alive_pids = {p.pid for p in psutil.process_iter(attrs=[])}
            for pid in list(self.store.keys()):
                # If process is dead and no samples remain
                if pid not in alive_pids and len(self.store[pid]) == 0:
                    del self.store[pid]

def sample_once(store: ProcessStore):
    """
    Collects a single set of metrics for all running processes.
    """
    now = time.time()
    for proc in psutil.process_iter(attrs=['pid', 'name', 'username', 'cmdline']):
        try:
            pid = proc.info['pid']
            cpu_percent = proc.cpu_percent(interval=None) # Non-blocking CPU usage
            mem_info = proc.memory_info()
            mem_percent = proc.memory_percent()
            # Safely fetch IO and file info
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
            continue # Ignore transient errors
        except Exception:
            traceback.print_exc()
            continue

def start_sampling(store: ProcessStore, interval=SAMPLE_INTERVAL):
    """
    Starts a background thread to continuously collect samples.
    """
    def run():
        while True:
            sample_once(store)
            time.sleep(interval)
            
    # Daemon thread ensures it closes when the main program exits
    t = threading.Thread(target=run, daemon=True) 
    t.start()
    return t
