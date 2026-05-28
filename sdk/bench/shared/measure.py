import os
import time
import resource
import threading
from collections.abc import Callable

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _cuda_sync() -> None:
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


class _Sampler(threading.Thread):
    _handle: object | None
    _alloc_probe: Callable[[], int] | None
    _interval: float
    _stop_event: threading.Event
    peak_nvml_bytes: int
    peak_alloc_bytes: int

    def __init__(self, handle: object | None,
                 alloc_probe: Callable[[], int] | None = None,
                 interval: float = 0.02) -> None:
        super().__init__(daemon=True)
        self._handle = handle
        self._alloc_probe = alloc_probe
        self._interval = interval
        self._stop_event = threading.Event()
        self.peak_nvml_bytes = 0
        self.peak_alloc_bytes = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._handle is not None:
                used = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
                if used > self.peak_nvml_bytes:
                    self.peak_nvml_bytes = used
            if self._alloc_probe is not None:
                alloc = self._alloc_probe()
                if alloc > self.peak_alloc_bytes:
                    self.peak_alloc_bytes = alloc
            self._stop_event.wait(self._interval)

    def stop(self) -> None:
        self._stop_event.set()


class Timer:
    _t0: float
    elapsed_s: float

    def __enter__(self) -> "Timer":
        _cuda_sync()
        self._t0 = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *exc) -> bool:
        _cuda_sync()
        self.elapsed_s = time.perf_counter() - self._t0
        return False


class Measure:
    _gpu_index: int
    _alloc_probe: Callable[[], int] | None
    _baseline_bytes: int
    _sampler: _Sampler | None
    _t0: float
    elapsed_s: float
    peak_vram_mb: float
    peak_alloc_mb: float
    peak_rss_mb: float

    def __init__(self, gpu_index: int = 0,
                 alloc_probe: Callable[[], int] | None = None) -> None:
        self._gpu_index = gpu_index
        self._alloc_probe = alloc_probe
        self._baseline_bytes = int(os.environ.get("BENCH_VRAM_BASELINE_BYTES") or 0)
        self._sampler = None

    def __enter__(self) -> "Measure":
        handle = None
        if _HAS_NVML:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        self._sampler = _Sampler(handle, self._alloc_probe)
        self._sampler.start()
        _cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        _cuda_sync()
        self.elapsed_s = time.perf_counter() - self._t0
        self._sampler.stop()
        self._sampler.join()
        vram_bytes = max(0, self._sampler.peak_nvml_bytes - self._baseline_bytes)
        self.peak_vram_mb = vram_bytes / 1024 ** 2
        self.peak_alloc_mb = self._sampler.peak_alloc_bytes / 1024 ** 2
        self.peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return False
