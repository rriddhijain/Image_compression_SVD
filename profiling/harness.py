"""
Accurate CUDA / Hardware Timing Profiling Harness.
Person B: Hardware Systems Lead.

Eliminates naive host timers for GPU workloads. Implements isolated execution timing:
- CUDA: torch.cuda.Event(enable_timing=True) with explicit stream synchronization.
- MPS/CPU: torch.mps.synchronize() with high-precision perf_counter_ns().
- 100+ warm-up iterations to reach clock stability and eliminate JIT / caching jitter.
- 200-500 timed iterations capturing Mean, P50 (median), P90, P99, and Standard Deviation.
- Peak VRAM tracking via memory.py.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from profiling.memory import MemoryTracker, get_model_size_mb


class BenchmarkHarness:
    """
    Standardized, jitter-free benchmarking engine for neural network inference.
    """
    def __init__(
        self,
        device: torch.device | None = None,
        warmup_iters: int = 100,
        active_iters: int = 300
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.warmup_iters = warmup_iters
        self.active_iters = active_iters
        self.is_cuda = (self.device.type == "cuda")
        self.is_mps = (self.device.type == "mps")
        self.mem_tracker = MemoryTracker(self.device)

    def synchronize(self):
        """Forces device synchronization across all hardware backends."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elif self.is_mps:
            torch.mps.synchronize()

    @torch.no_grad()
    def profile_layer(
        self,
        layer: nn.Module,
        input_tensor: torch.Tensor
    ) -> dict:
        """
        Profiles a single layer (e.g. nn.Linear or DecomposedLinear) in isolation.
        """
        layer.eval()
        layer.to(self.device)
        x = input_tensor.to(self.device)

        # 1. Warm-up Phase: eliminate driver caching, instruction cache misses, power state ramps
        for _ in range(self.warmup_iters):
            _ = layer(x)
        self.synchronize()

        # 2. Reset memory tracker
        self.mem_tracker.reset()

        # 3. Active timing
        latencies_ms = []

        if self.is_cuda:
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_iters)]

            for i in range(self.active_iters):
                start_events[i].record()
                _ = layer(x)
                end_events[i].record()

            self.synchronize()

            for i in range(self.active_iters):
                latencies_ms.append(start_events[i].elapsed_time(end_events[i]))
        else:
            # High-precision timer for MPS/CPU with explicit stream sync
            for _ in range(self.active_iters):
                self.synchronize()
                t0 = time.perf_counter_ns()
                _ = layer(x)
                self.synchronize()
                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1e6)  # convert ns to ms

        lat_arr = np.array(latencies_ms)
        mem_stats = self.mem_tracker.get_stats()

        return {
            "device": str(self.device),
            "batch_size": input_tensor.shape[0],
            "input_dim": input_tensor.shape[1],
            "param_size_mb": get_model_size_mb(layer),
            "mean_ms": float(np.mean(lat_arr)),
            "std_ms": float(np.std(lat_arr)),
            "p50_ms": float(np.percentile(lat_arr, 50)),
            "p90_ms": float(np.percentile(lat_arr, 90)),
            "p99_ms": float(np.percentile(lat_arr, 99)),
            "throughput_samples_per_sec": float((input_tensor.shape[0] * 1000.0) / np.mean(lat_arr)),
            "peak_vram_mb": mem_stats["max_allocated_mb"]
        }

    @torch.no_grad()
    def profile_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor
    ) -> dict:
        """
        Profiles end-to-end forward inference of a complete neural network.
        """
        model.eval()
        model.to(self.device)
        x = sample_input.to(self.device)

        # Warm-up
        for _ in range(self.warmup_iters):
            _ = model(x)
        self.synchronize()

        self.mem_tracker.reset()

        latencies_ms = []

        if self.is_cuda:
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_iters)]

            for i in range(self.active_iters):
                start_events[i].record()
                _ = model(x)
                end_events[i].record()

            self.synchronize()

            for i in range(self.active_iters):
                latencies_ms.append(start_events[i].elapsed_time(end_events[i]))
        else:
            for _ in range(self.active_iters):
                self.synchronize()
                t0 = time.perf_counter_ns()
                _ = model(x)
                self.synchronize()
                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1e6)

        lat_arr = np.array(latencies_ms)
        mem_stats = self.mem_tracker.get_stats()

        return {
            "device": str(self.device),
            "batch_size": sample_input.shape[0],
            "param_size_mb": get_model_size_mb(model),
            "mean_ms": float(np.mean(lat_arr)),
            "std_ms": float(np.std(lat_arr)),
            "p50_ms": float(np.percentile(lat_arr, 50)),
            "p90_ms": float(np.percentile(lat_arr, 90)),
            "p99_ms": float(np.percentile(lat_arr, 99)),
            "throughput_samples_per_sec": float((sample_input.shape[0] * 1000.0) / np.mean(lat_arr)),
            "peak_vram_mb": mem_stats["max_allocated_mb"]
        }
