# utils/generate_load.py
import time, threading

def cpu_stress(duration=10, threads=2):
    def worker():
        end = time.time() + duration
        while time.time() < end:
            x = 0
            for i in range(10000):
                x += i*i
    for _ in range(threads):
        t = threading.Thread(target=worker)
        t.start()

if __name__ == "__main__":
    print("Starting CPU stress for 30s")
    cpu_stress(duration=30, threads=4)
    print("Done")
