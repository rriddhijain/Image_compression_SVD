"""
Publication-Quality Pareto and Roofline Visualizations.
Person B: Hardware Systems Lead.

Generates:
1. Plot 1: Compression Ratio vs. Accuracy Retention (marking the optimal Pareto knee point).
2. Plot 2: Theoretical FLOPs vs. Measured Wall-Clock Latency (revealing latency stagnation).
3. Plot 3: Empirical Roofline Model (mapping operational intensity across batch sizes).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set high-quality styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})


def plot_pareto_compression_vs_accuracy(
    df: pd.DataFrame,
    output_path: str = "results/plots/pareto_compression_vs_accuracy.png"
):
    """
    Plot 1: Compression Ratio vs. Accuracy Retention
    Identifies the Pareto knee point balancing parameter reduction vs. accuracy.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Baseline reference
    baseline_row = df[df["is_decomposed"] == False]
    if not baseline_row.empty:
        base_acc = baseline_row["accuracy"].iloc[0] * 100
        ax.axhline(base_acc, color="#333333", linestyle=":", label=f"Uncompressed Baseline ({base_acc:.1f}%)")

    decomp_df = df[df["is_decomposed"] == True].sort_values("rank")

    if "finetuned_acc" in decomp_df.columns:
        ax.plot(
            decomp_df["compression_ratio"],
            decomp_df["finetuned_acc"] * 100,
            "o-",
            color="#2ca02c",
            label="Fine-tuned (Sainath et al.)",
            markersize=8
        )
    
    if "zero_shot_acc" in decomp_df.columns:
        ax.plot(
            decomp_df["compression_ratio"],
            decomp_df["zero_shot_acc"] * 100,
            "s--",
            color="#d62728",
            label="Zero-shot (Post-SVD, Un-tuned)",
            markersize=7
        )

    # Annotate ranks
    for _, row in decomp_df.iterrows():
        rank = int(row["rank"])
        cr = row["compression_ratio"]
        acc = row.get("finetuned_acc", row.get("accuracy", 0)) * 100
        ax.annotate(
            f"k={rank}",
            (cr, acc),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xlabel("Compression Ratio (Relative to Baseline)", fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax.set_title("Pareto Frontier: Parameter Compression vs. Accuracy Retention", fontweight="bold")
    ax.grid(True)
    ax.legend(frameon=True, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")


def plot_flops_vs_latency_stagnation(
    df: pd.DataFrame,
    output_path: str = "results/plots/theoretical_flops_vs_measured_latency.png"
):
    """
    Plot 2: Theoretical FLOPs vs. Measured Wall-Clock Latency.
    Exposes the divergence where low FLOPs do not translate into speedups at B=1.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    batch_sizes = sorted(df["batch_size"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))

    # Subplot A: Latency vs Rank across Batch Sizes
    for bsz, color in zip(batch_sizes, colors):
        subset = df[df["batch_size"] == bsz].sort_values("rank", na_position="first")
        
        # Decomposed subset
        decomp = subset[subset["is_decomposed"] == True]
        base = subset[subset["is_decomposed"] == False]
        
        line, = ax1.plot(
            decomp["rank"],
            decomp["p50_ms"],
            "o-",
            label=f"B_sz = {bsz}",
            color=color,
            markersize=6
        )
        if not base.empty:
            ax1.axhline(base["p50_ms"].iloc[0], color=color, linestyle=":", alpha=0.6)

    ax1.set_xlabel("Truncated Rank $k$", fontweight="bold")
    ax1.set_ylabel("P50 Wall-Clock Latency (ms)", fontweight="bold")
    ax1.set_title("Empirical Latency vs. Rank by Batch Size", fontweight="bold")
    ax1.grid(True)
    ax1.legend(title="Batch Size", frameon=True)

    # Subplot B: Theoretical Speedup vs Measured Speedup (Highlighting Memory Ceiling at B=1)
    b1_subset = df[df["batch_size"] == 1].sort_values("rank")
    b1_base = b1_subset[b1_subset["is_decomposed"] == False]
    b1_decomp = b1_subset[b1_subset["is_decomposed"] == True]

    if not b1_base.empty and not b1_decomp.empty:
        base_flops = b1_base["theoretical_flops"].iloc[0]
        base_lat = b1_base["p50_ms"].iloc[0]

        theoretical_speedup = base_flops / b1_decomp["theoretical_flops"]
        measured_speedup = base_lat / b1_decomp["p50_ms"]

        ax2.plot(
            b1_decomp["rank"],
            theoretical_speedup,
            "k--",
            label="Theoretical FLOP Speedup (Ideal)",
            lw=2.2
        )
        ax2.plot(
            b1_decomp["rank"],
            measured_speedup,
            "r-o",
            label="Empirical Wall-Clock Speedup (B=1)",
            lw=2.2,
            markersize=7
        )
        ax2.axhline(1.0, color="gray", linestyle=":", label="1.0x Parity")

        ax2.set_xlabel("Truncated Rank $k$", fontweight="bold")
        ax2.set_ylabel("Speedup over Uncompressed Baseline", fontweight="bold")
        ax2.set_title("The Memory Bandwidth Wall (B=1):\nFLOP Reduction vs. Empirical Speedup", fontweight="bold")
        ax2.grid(True)
        ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")


def plot_empirical_roofline(
    df: pd.DataFrame,
    peak_gflops: float = 2000.0,
    peak_bandwidth_gb_s: float = 100.0,
    output_path: str = "results/plots/empirical_roofline.png"
):
    """
    Plot 3: Empirical Roofline Model (Williams et al., 2009).
    Maps operational intensity (FLOPs/Byte) to measured GFLOP/s against hardware ceiling.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Compute knee intensity where compute ceiling intersects bandwidth ceiling
    # Peak_GFLOPs / (Peak_BW_GB/s) = GFLOP / Byte = FLOP / Byte
    knee_intensity = peak_gflops / peak_bandwidth_gb_s

    # Roofline boundary curve
    x_intensities = np.logspace(-2, 3, 400)
    y_attainable = np.minimum(peak_gflops, peak_bandwidth_gb_s * x_intensities)

    ax.loglog(x_intensities, y_attainable, "k-", lw=2.5, label="Hardware Roofline Boundary")
    ax.axvline(knee_intensity, color="gray", linestyle=":", alpha=0.7, label=f"Knee Intensity ({knee_intensity:.2f} FLOP/B)")

    # Color by batch size
    batch_sizes = sorted(df["batch_size"].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(batch_sizes)))

    for bsz, color in zip(batch_sizes, colors):
        subset = df[df["batch_size"] == bsz]
        # Empirical GFLOP/s = (theoretical_flops / (latency_ms / 1000)) / 1e9
        empirical_gflops = (subset["theoretical_flops"] / (subset["p50_ms"] / 1000.0)) / 1e9
        
        ax.scatter(
            subset["arithmetic_intensity"],
            empirical_gflops,
            color=color,
            s=65,
            edgecolors="black",
            alpha=0.85,
            label=f"B_sz = {bsz}"
        )

    ax.set_xlabel(r"Operational Arithmetic Intensity $\frac{\text{FLOPs}}{\text{Byte transferred}}$ (log scale)", fontweight="bold")
    ax.set_ylabel(r"Attainable Performance (GFLOP/s, log scale)", fontweight="bold")
    ax.set_title("Empirical Roofline Model (Williams et al., 2009)\nMemory Bandwidth Bound vs. Compute Bound Regimes", fontweight="bold")
    ax.grid(True, which="both", ls="--")
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")
