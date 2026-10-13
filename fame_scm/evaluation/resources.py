"""Measured process CPU and sampled process-tree RSS, with explicit scope."""

import threading
import time
import psutil


class ResourceMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0
        self.stop_event = threading.Event()
        self.sampling_error = None

    def start(self):
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        try:
            process = psutil.Process()
        except psutil.Error as exc:
            self.sampling_error = str(exc)
            return
        while not self.stop_event.is_set():
            try:
                self.peak = max(
                    self.peak,
                    sum(
                        p.memory_info().rss
                        for p in [process] + process.children(recursive=True)
                    ),
                )
            except psutil.Error as exc:
                self.sampling_error = str(exc)
            self.stop_event.wait(self.interval)

    def finish(self):
        if hasattr(self, "result"):
            return self.result
        self.stop_event.set()
        self.thread.join()
        self.result = {
            "wall_seconds": time.perf_counter() - self.wall,
            "cpu_process_seconds": time.process_time() - self.cpu,
            "cpu_process_hours": (time.process_time() - self.cpu) / 3600,
            "peak_process_tree_rss_bytes": self.peak or None,
            "memory_sampling_error": self.sampling_error,
            "sampling_interval_seconds": self.interval,
            "gpu_resource_hours": None,
            "scope": "Whole selected method including test/audit; CPU excludes child-process CPU; RSS is sampled. Stage wall times are separately reported. GPU hours require external telemetry.",
        }
        return self.result
