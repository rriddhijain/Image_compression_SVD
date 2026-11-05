"""
Analytical Compute & Operational Arithmetic Intensity Module.
Person B: Hardware Systems Lead.

Provides exact FLOP/MAC counters, breakeven formulas, and Roofline metrics
(Williams et al., 2009) for standard and decomposed linear layers.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class LayerComplexity:
    layer_name: str
    m_in: int
    n_out: int
    rank: int | None
    batch_size: int
    params: int
    param_bytes: int
    theoretical_flops: int
    theoretical_macs: int
    activation_bytes: int
    total_memory_traffic_bytes: int
    arithmetic_intensity: float  # FLOPs / Byte
    is_decomposed: bool
    breakeven_rank: float | None = None


def compute_linear_flops(
    m_in: int,
    n_out: int,
    batch_size: int = 1,
    has_bias: bool = True
) -> int:
    """
    Standard GEMM FLOPs for Y = X @ W^T + b:
    - Matrix multiplication: 2 * B_sz * M_in * N_out (multiply + accumulate)
    - Bias addition: B_sz * N_out (if has_bias)
    """
    flops = 2 * batch_size * m_in * n_out
    if has_bias:
        flops += batch_size * n_out
    return flops


def compute_decomposed_flops(
    m_in: int,
    n_out: int,
    rank: int,
    batch_size: int = 1,
    has_bias: bool = True
) -> int:
    """
    Two-stage cascaded GEMM FLOPs:
    Stage 1: H = X @ L1^T (shape [B, K]), no bias
             FLOPs = 2 * B_sz * M_in * K
    Stage 2: Y = H @ L2^T + b (shape [B, N]), with bias
             FLOPs = 2 * B_sz * K * N_out + (B_sz * N_out if has_bias else 0)
    Total FLOPs = 2 * B_sz * K * (M_in + N_out) + (B_sz * N_out if has_bias else 0)
    """
    flops = 2 * batch_size * rank * (m_in + n_out)
    if has_bias:
        flops += batch_size * n_out
    return flops


def compute_breakeven_rank(m_in: int, n_out: int) -> float:
    """
    Rank K* at which decomposed FLOPs equal standard linear FLOPs:
    2 * B * K* * (M + N) = 2 * B * M * N  =>  K* = (M * N) / (M + N)
    """
    return (m_in * n_out) / (m_in + n_out)


def calculate_arithmetic_intensity(
    m_in: int,
    n_out: int,
    rank: int | None = None,
    batch_size: int = 1,
    bytes_per_elem: int = 4,  # FP32 = 4 bytes, FP16/BF16 = 2 bytes
    has_bias: bool = True
) -> LayerComplexity:
    """
    Calculates operational arithmetic intensity (FLOPs / Byte transferred).
    Per Williams et al. (2009):
    Arithmetic Intensity = Total FLOPs / Total Memory Traffic (Bytes)
    
    Memory Traffic in an un-fused layer execution:
    - Standard Layer:
        Weights: M * N + (N if bias) elements read
        Input activation: B * M elements read
        Output activation: B * N elements written
    - Decomposed Layer:
        Weights L1: M * K elements read
        Weights L2: K * N + (N if bias) elements read
        Input activation: B * M elements read
        Intermediate activation: B * K elements written to VRAM + B * K elements read from VRAM
        Output activation: B * N elements written
    """
    if rank is None or rank <= 0:
        # Standard Linear Layer
        params = m_in * n_out + (n_out if has_bias else 0)
        param_bytes = params * bytes_per_elem
        
        flops = compute_linear_flops(m_in, n_out, batch_size=batch_size, has_bias=has_bias)
        macs = batch_size * m_in * n_out

        # Memory traffic
        input_act_bytes = batch_size * m_in * bytes_per_elem
        output_act_bytes = batch_size * n_out * bytes_per_elem
        act_traffic_bytes = input_act_bytes + output_act_bytes
        total_traffic = param_bytes + act_traffic_bytes
        ai = flops / total_traffic if total_traffic > 0 else 0.0

        return LayerComplexity(
            layer_name="StandardLinear",
            m_in=m_in,
            n_out=n_out,
            rank=None,
            batch_size=batch_size,
            params=params,
            param_bytes=param_bytes,
            theoretical_flops=flops,
            theoretical_macs=macs,
            activation_bytes=act_traffic_bytes,
            total_memory_traffic_bytes=total_traffic,
            arithmetic_intensity=ai,
            is_decomposed=False,
            breakeven_rank=compute_breakeven_rank(m_in, n_out)
        )
    else:
        # Decomposed Pair (L1: M -> K, L2: K -> N)
        params_l1 = m_in * rank
        params_l2 = rank * n_out + (n_out if has_bias else 0)
        total_params = params_l1 + params_l2
        param_bytes = total_params * bytes_per_elem

        flops = compute_decomposed_flops(m_in, n_out, rank, batch_size=batch_size, has_bias=has_bias)
        macs = batch_size * rank * (m_in + n_out)

        # Activations: Input read + intermediate round-trip (write L1, read L2) + output write
        input_act_bytes = batch_size * m_in * bytes_per_elem
        intermediate_act_roundtrip_bytes = 2 * batch_size * rank * bytes_per_elem
        output_act_bytes = batch_size * n_out * bytes_per_elem
        act_traffic_bytes = input_act_bytes + intermediate_act_roundtrip_bytes + output_act_bytes

        total_traffic = param_bytes + act_traffic_bytes
        ai = flops / total_traffic if total_traffic > 0 else 0.0

        return LayerComplexity(
            layer_name=f"DecomposedLinear_k{rank}",
            m_in=m_in,
            n_out=n_out,
            rank=rank,
            batch_size=batch_size,
            params=total_params,
            param_bytes=param_bytes,
            theoretical_flops=flops,
            theoretical_macs=macs,
            activation_bytes=act_traffic_bytes,
            total_memory_traffic_bytes=total_traffic,
            arithmetic_intensity=ai,
            is_decomposed=True,
            breakeven_rank=compute_breakeven_rank(m_in, n_out)
        )


def model_theoretical_summary(model: nn.Module, batch_size: int = 1) -> dict:
    """
    Computes total parameter count and theoretical FLOPs across all linear layers in a PyTorch module.
    """
    total_flops = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            m_in, n_out = module.in_features, module.out_features
            has_bias = module.bias is not None
            flops = compute_linear_flops(m_in, n_out, batch_size, has_bias)
            total_flops += flops
            total_params += sum(p.numel() for p in module.parameters())
            
    return {
        "batch_size": batch_size,
        "total_params": total_params,
        "theoretical_linear_flops": total_flops,
        "param_memory_mb_fp32": (total_params * 4) / (1024 ** 2)
    }
