"""
VRAM and Memory Allocation Tracking Module.
Person B: Hardware Systems Lead.

Provides exact tracking of model parameter footprints, static state,
and peak transient activation footprints during forward passes.
"""

import os
import torch
import torch.nn as nn


def get_model_size_mb(model: nn.Module) -> float:
    """
    Computes total memory size of all parameters and buffers in megabytes (MB).
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.numel() * buf.element_size()
    return total_bytes / (1024.0 * 1024.0)


def get_checkpoint_disk_size_mb(filepath: str) -> float:
    """
    Returns the physical file size of a saved PyTorch checkpoint on disk.
    """
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024.0 * 1024.0)
    return 0.0


class MemoryTracker:
    """
    Instruments memory during forward execution.
    Supports CUDA (with hardware allocation tracking) and fallback for MPS/CPU.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = (device.type == "cuda")

    def reset(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

    def get_stats(self) -> dict:
        if self.is_cuda:
            return {
                "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0),
                "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0),
                "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0),
                "max_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0)
            }
        else:
            return {
                "allocated_mb": 0.0,
                "max_allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_reserved_mb": 0.0
            }
