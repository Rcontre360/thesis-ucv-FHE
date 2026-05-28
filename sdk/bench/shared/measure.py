import os
import time
import resource
import threading
from collections.abc import Callable

import pynvml
import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def phase_metrics(phases: dict[str, "Measure | None"]) -> dict:
    """Flatten ordered named Measure phases into prefixed result columns.

    `phases` is insertion-ordered {name: Measure}. Per phase emits
    `<name>_{vram,vram_alloc}_{mb,delta_mb}` and `<name>_ram_mb`, where each
    delta is this phase's peak minus the previous phase's peak (0 for the first).
    A `None` phase (one the backend does not run) emits zeros and is skipped
    when computing the next phase's delta.
    """
    out: dict = {}
    prev: Measure | None = None
    for name, m in phases.items():
        if m is None:
            out[f"{name}_vram_mb"] = out[f"{name}_vram_delta_mb"] = 0.0
            out[f"{name}_vram_alloc_mb"] = out[f"{name}_vram_alloc_delta_mb"] = 0.0
            out[f"{name}_ram_mb"] = 0.0
            continue
        out[f"{name}_vram_mb"] = m.peak_vram_mb
        out[f"{name}_vram_delta_mb"] = 0.0 if prev is None else m.peak_vram_mb - prev.peak_vram_mb
        out[f"{name}_vram_alloc_mb"] = m.peak_alloc_mb
        out[f"{name}_vram_alloc_delta_mb"] = 0.0 if prev is None else m.peak_alloc_mb - prev.peak_alloc_mb
        out[f"{name}_ram_mb"] = m.peak_rss_mb
        prev = m
    return out


class _Sampler(threading.Thread):
    _handle: object                              # NVML device handle to poll
    _alloc_probe: Callable[[], int] | None       # optional: per-backend allocator counter (live bytes)
    _interval: float                             # poll period in seconds
    _stop_event: threading.Event                 # signaled by stop() to break the polling loop
    peak_nvml_bytes: int                         # max NVML "used" seen during the run (device-wide footprint)
    peak_alloc_bytes: int                        # max alloc_probe() value seen during the run (working set)

    def __init__(self, handle: object,
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
            self.peak_nvml_bytes = max(self.peak_nvml_bytes, pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            if self._alloc_probe is not None:
                self.peak_alloc_bytes = max(self.peak_alloc_bytes, self._alloc_probe())
            self._stop_event.wait(self._interval)

    def stop(self) -> None:
        self._stop_event.set()


class Timer:
    _t0: float                                   # perf_counter() reading captured on __enter__
    elapsed_s: float                             # GPU-synced wall time between __enter__ and __exit__

    def __enter__(self) -> "Timer":
        cuda_sync()
        self._t0 = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *exc) -> bool:
        cuda_sync()
        self.elapsed_s = time.perf_counter() - self._t0
        return False


class Measure:
    _gpu_index: int                              # CUDA device index passed to NVML
    _alloc_probe: Callable[[], int] | None       # per-backend "live bytes now" probe (e.g. RMM pool counter)
    _baseline_bytes: int                         # pre-benchmark device-wide VRAM floor subtracted from peak_vram_mb
    _sampler: _Sampler | None                    # background polling thread, alive only between enter/exit
    _t0: float                                   # perf_counter() reading captured on __enter__

    elapsed_s: float                             # GPU-synced wall time between __enter__ and __exit__
    peak_vram_mb: float                          # peak NVML footprint minus baseline (MB)
    peak_alloc_mb: float                         # peak working-set bytes from alloc_probe (MB; 0 if no probe)
    peak_rss_mb: float                           # peak CPU RSS via ru_maxrss (MB)

    def __init__(self, gpu_index: int = 0,
                 alloc_probe: Callable[[], int] | None = None) -> None:
        self._gpu_index = gpu_index
        self._alloc_probe = alloc_probe
        self._baseline_bytes = int(os.environ.get("BENCH_VRAM_BASELINE_BYTES") or 0)
        self._sampler = None

    def __enter__(self) -> "Measure":
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        self._sampler = _Sampler(handle, self._alloc_probe)
        self._sampler.start()
        cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        cuda_sync()
        self.elapsed_s = time.perf_counter() - self._t0
        self._sampler.stop()
        self._sampler.join()
        vram_bytes = max(0, self._sampler.peak_nvml_bytes - self._baseline_bytes)
        self.peak_vram_mb = vram_bytes / 1024 ** 2
        self.peak_alloc_mb = self._sampler.peak_alloc_bytes / 1024 ** 2
        self.peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return False
