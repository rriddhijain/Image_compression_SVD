"""
Systematic Experimental Benchmarking Sweep.
Person B: Hardware Systems Lead.

Runs multi-variable parameter sweeps across:
- Truncated Ranks: k in [8, 16, 32, 64, 128, 256]
- Batch Sizes: B_sz in [1, 8, 32, 64, 128]
- Layerwise isolation vs. end-to-end model inference

Outputs raw structured metrics to results/benchmark_data.csv.
Automatically detects CUDA, MPS, or CPU hardware backend.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pandas as pd
import torch
import torch.nn as nn

from models.baseline import BaselineMLP
from models.decomposed import DecomposedLinear, decompose_model_linear_layers
from profiling.harness import BenchmarkHarness
from profiling.metrics import (
    calculate_arithmetic_intensity,
    compute_linear_flops,
    compute_decomposed_flops
)
from profiling.memory import get_model_size_mb
from visualizations.pareto import (
    plot_flops_vs_latency_stagnation,
    plot_empirical_roofline,
    plot_pareto_compression_vs_accuracy
)


def run_layerwise_benchmark(
    in_features: int = 1024,
    out_features: int = 512,
    ranks: list[int] = [8, 16, 32, 64, 128, 256],
    batch_sizes: list[int] = [1, 8, 32, 64, 128],
    warmup_iters: int = 50,
    active_iters: int = 200,
    device_name: str | None = None
) -> pd.DataFrame:
    """
    Benchmarks isolated GEMM kernels: Standard nn.Linear vs. DecomposedLinear.
    """
    if device_name:
        device = torch.device(device_name)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"\n=======================================================")
    print(f"[*] Initializing Isolated Layer Benchmarks on {device}")
    print(f"[*] Matrix: {in_features} -> {out_features}")
    print(f"=======================================================")

    harness = BenchmarkHarness(device=device, warmup_iters=warmup_iters, active_iters=active_iters)
    records = []

    # 1. Benchmark Standard Layer across all batch sizes
    orig_linear = nn.Linear(in_features, out_features, bias=True).to(device)

    for bsz in batch_sizes:
        x = torch.randn(bsz, in_features, device=device)
        timing_stats = harness.profile_layer(orig_linear, x)
        ai_stats = calculate_arithmetic_intensity(in_features, out_features, rank=None, batch_size=bsz)

        records.append({
            "experiment_type": "isolated_layer",
            "layer_name": "StandardLinear",
            "is_decomposed": False,
            "rank": None,
            "batch_size": bsz,
            "in_features": in_features,
            "out_features": out_features,
            "param_count": ai_stats.params,
            "param_size_mb": timing_stats["param_size_mb"],
            "compression_ratio": 1.0,
            "theoretical_flops": ai_stats.theoretical_flops,
            "theoretical_macs": ai_stats.theoretical_macs,
            "arithmetic_intensity": ai_stats.arithmetic_intensity,
            "total_mem_traffic_bytes": ai_stats.total_memory_traffic_bytes,
            "mean_ms": timing_stats["mean_ms"],
            "std_ms": timing_stats["std_ms"],
            "p50_ms": timing_stats["p50_ms"],
            "p90_ms": timing_stats["p90_ms"],
            "p99_ms": timing_stats["p99_ms"],
            "throughput_samples_sec": timing_stats["throughput_samples_per_sec"],
            "peak_vram_mb": timing_stats["peak_vram_mb"],
            "device": str(device)
        })
        print(f"Baseline | B={bsz:<3} | P50: {timing_stats['p50_ms']:.4f} ms | AI: {ai_stats.arithmetic_intensity:.2f} FLOP/B")

    # 2. Benchmark Decomposed Linear across all ranks and batch sizes
    baseline_params = in_features * out_features + out_features

    for k in ranks:
        decomp_layer = DecomposedLinear.from_linear(orig_linear, rank=k).to(device)
        decomp_params = in_features * k + k * out_features + out_features
        cr = baseline_params / decomp_params

        for bsz in batch_sizes:
            x = torch.randn(bsz, in_features, device=device)
            timing_stats = harness.profile_layer(decomp_layer, x)
            ai_stats = calculate_arithmetic_intensity(in_features, out_features, rank=k, batch_size=bsz)

            records.append({
                "experiment_type": "isolated_layer",
                "layer_name": f"DecomposedLinear_k{k}",
                "is_decomposed": True,
                "rank": k,
                "batch_size": bsz,
                "in_features": in_features,
                "out_features": out_features,
                "param_count": decomp_params,
                "param_size_mb": timing_stats["param_size_mb"],
                "compression_ratio": cr,
                "theoretical_flops": ai_stats.theoretical_flops,
                "theoretical_macs": ai_stats.theoretical_macs,
                "arithmetic_intensity": ai_stats.arithmetic_intensity,
                "total_mem_traffic_bytes": ai_stats.total_memory_traffic_bytes,
                "mean_ms": timing_stats["mean_ms"],
                "std_ms": timing_stats["std_ms"],
                "p50_ms": timing_stats["p50_ms"],
                "p90_ms": timing_stats["p90_ms"],
                "p99_ms": timing_stats["p99_ms"],
                "throughput_samples_sec": timing_stats["throughput_samples_per_sec"],
                "peak_vram_mb": timing_stats["peak_vram_mb"],
                "device": str(device)
            })
            print(f"Rank {k:<3}  | B={bsz:<3} | P50: {timing_stats['p50_ms']:.4f} ms | AI: {ai_stats.arithmetic_intensity:.2f} FLOP/B | CR: {cr:.2f}x")

    return pd.DataFrame(records)


def run_end_to_end_benchmark(
    ranks: list[int] = [8, 16, 32, 64, 128, 256],
    batch_sizes: list[int] = [1, 8, 32, 64, 128],
    warmup_iters: int = 50,
    active_iters: int = 150,
    device_name: str | None = None
) -> pd.DataFrame:
    """
    Benchmarks full end-to-end MLP inference.
    """
    if device_name:
        device = torch.device(device_name)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"\n=======================================================")
    print(f"[*] Initializing End-to-End Model Benchmarks on {device}")
    print(f"=======================================================")

    harness = BenchmarkHarness(device=device, warmup_iters=warmup_iters, active_iters=active_iters)
    records = []

    base_model = BaselineMLP(in_features=3072, num_classes=10).to(device)
    base_params = sum(p.numel() for p in base_model.parameters())

    # Baseline Model across batches
    for bsz in batch_sizes:
        x = torch.randn(bsz, 3072, device=device)
        timing_stats = harness.profile_model(base_model, x)
        
        # Calculate approximate linear FLOPs
        # 3072->1024, 1024->512, 512->256, 256->10
        flops = (
            compute_linear_flops(3072, 1024, bsz) +
            compute_linear_flops(1024, 512, bsz) +
            compute_linear_flops(512, 256, bsz) +
            compute_linear_flops(256, 10, bsz)
        )
        total_param_bytes = base_params * 4
        total_act_traffic = bsz * (3072 + 2*1024 + 2*512 + 2*256 + 10) * 4
        ai = flops / (total_param_bytes + total_act_traffic)

        records.append({
            "experiment_type": "end_to_end_model",
            "model_name": "BaselineMLP",
            "is_decomposed": False,
            "rank": None,
            "batch_size": bsz,
            "param_count": base_params,
            "param_size_mb": timing_stats["param_size_mb"],
            "compression_ratio": 1.0,
            "theoretical_flops": flops,
            "arithmetic_intensity": ai,
            "mean_ms": timing_stats["mean_ms"],
            "std_ms": timing_stats["std_ms"],
            "p50_ms": timing_stats["p50_ms"],
            "p90_ms": timing_stats["p90_ms"],
            "p99_ms": timing_stats["p99_ms"],
            "throughput_samples_sec": timing_stats["throughput_samples_per_sec"],
            "peak_vram_mb": timing_stats["peak_vram_mb"],
            "device": str(device)
        })

    # Decomposed Models
    for k in ranks:
        decomp_model = decompose_model_linear_layers(base_model, rank=k).to(device)
        decomp_params = sum(p.numel() for p in decomp_model.parameters())
        cr = base_params / decomp_params

        for bsz in batch_sizes:
            x = torch.randn(bsz, 3072, device=device)
            timing_stats = harness.profile_model(decomp_model, x)

            flops = (
                compute_decomposed_flops(3072, 1024, k, bsz) +
                compute_decomposed_flops(1024, 512, min(k, 512), bsz) +
                compute_decomposed_flops(512, 256, min(k, 256), bsz) +
                compute_linear_flops(256, 10, bsz)
            )
            total_param_bytes = decomp_params * 4
            # Accounting for intermediate activation buffers
            total_act_traffic = bsz * (
                3072 + 2*k + 2*1024 + 2*min(k, 512) + 2*512 + 2*min(k, 256) + 2*256 + 10
            ) * 4
            ai = flops / (total_param_bytes + total_act_traffic)

            records.append({
                "experiment_type": "end_to_end_model",
                "model_name": f"DecomposedMLP_k{k}",
                "is_decomposed": True,
                "rank": k,
                "batch_size": bsz,
                "param_count": decomp_params,
                "param_size_mb": timing_stats["param_size_mb"],
                "compression_ratio": cr,
                "theoretical_flops": flops,
                "arithmetic_intensity": ai,
                "mean_ms": timing_stats["mean_ms"],
                "std_ms": timing_stats["std_ms"],
                "p50_ms": timing_stats["p50_ms"],
                "p90_ms": timing_stats["p90_ms"],
                "p99_ms": timing_stats["p99_ms"],
                "throughput_samples_sec": timing_stats["throughput_samples_per_sec"],
                "peak_vram_mb": timing_stats["peak_vram_mb"],
                "device": str(device)
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Run SVD hardware profiling sweep.")
    parser.add_argument("--out_csv", type=str, default="results/benchmark_data.csv")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # 1. Run Isolated Layer Sweep
    layer_df = run_layerwise_benchmark(
        in_features=1024,
        out_features=512,
        ranks=[8, 16, 32, 64, 128, 256],
        batch_sizes=[1, 8, 32, 64, 128],
        warmup_iters=50,
        active_iters=150,
        device_name=args.device
    )

    # 2. Run End-to-End Model Sweep
    e2e_df = run_end_to_end_benchmark(
        ranks=[8, 16, 32, 64, 128, 256],
        batch_sizes=[1, 8, 32, 64, 128],
        warmup_iters=30,
        active_iters=100,
        device_name=args.device
    )

    combined_df = pd.concat([layer_df, e2e_df], ignore_index=True)
    combined_df.to_csv(args.out_csv, index=False)
    print(f"\n[+] Successfully saved {len(combined_df)} benchmark records to {args.out_csv}")

    # Generate Visualizations for isolated layer sweep
    plot_flops_vs_latency_stagnation(
        layer_df,
        output_path="results/plots/theoretical_flops_vs_measured_latency.png"
    )
    plot_empirical_roofline(
        layer_df,
        output_path="results/plots/empirical_roofline.png"
    )


if __name__ == "__main__":
    main()
