"""
Image SVD Intuition Builder (Phase 1).
Performs truncated SVD on a 2D grayscale image (512x512) across ranks:
k in {4, 8, 16, 32, 64, 128, 256}.

Quantifies:
1. Relative Frobenius Reconstruction Error: ||I - I_hat_k||_F / ||I||_F
2. Compression Ratio: (M * N) / (k * (M + N + 1))
3. Cumulative Singular Value Energy: sum(sigma_1..k) / sum(sigma_all)
4. Saves comparison figures and spectral decay plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_or_load_test_image(size: int = 512, filepath: str = "data/cameraman.png") -> np.ndarray:
    """
    Loads an image if present, or synthesizes a rich multi-frequency 2D benchmark
    matrix (geometric shapes + gradients + textured frequencies) of size (M, N).
    """
    if os.path.exists(filepath):
        img = Image.open(filepath).convert("L").resize((size, size))
        return np.asarray(img, dtype=np.float64) / 255.0

    # High quality synthetic multi-frequency benchmark image
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Combination of smooth low-frequency components + medium texture + sharp edges
    low_freq = np.sin(X) * np.cos(Y)
    radial = np.exp(-0.5 * (X**2 + Y**2))
    med_freq = 0.3 * np.sin(5 * X + 3 * Y)
    high_freq_stripes = 0.15 * np.sin(20 * X) * (np.abs(Y) < 1.5)
    geometric_disc = ((X - 0.8)**2 + (Y + 0.8)**2 < 0.6).astype(np.float64) * 0.7
    
    composite = low_freq + radial + med_freq + high_freq_stripes + geometric_disc
    # Normalize to [0, 1]
    composite = (composite - composite.min()) / (composite.max() - composite.min())
    
    # Save synthetic reference for reproducible viewing
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = Image.fromarray((composite * 255).astype(np.uint8))
    img.save(filepath)
    return composite


def run_image_svd(
    image: np.ndarray,
    ranks: list[int] = [4, 8, 16, 32, 64, 128, 256],
    output_dir: str = "results/plots"
) -> dict:
    """
    Decomposes the image using SVD, performs rank truncations, and records metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    M, N = image.shape
    min_dim = min(M, N)

    # Full SVD: I = U * diag(S) * Vt
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    frobenius_orig = np.linalg.norm(image, "fro")
    total_energy = np.sum(S)

    metrics = {
        "rank": [],
        "frob_error": [],
        "compression_ratio": [],
        "energy_ratio": [],
        "reconstructed_images": {}
    }

    for k in ranks:
        if k > min_dim:
            continue
        
        # Truncated reconstruction: I_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
        reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]
        
        # 1. Relative Frobenius Reconstruction Error
        err_fro = np.linalg.norm(image - reconstructed, "fro") / frobenius_orig
        
        # 2. Compression Ratio: (M * N) / (k * (M + N + 1))
        # Accounting for U (M*k), Vt (k*N), and singular values S (k)
        param_original = M * N
        param_compressed = k * (M + N + 1)
        cr = param_original / param_compressed
        
        # 3. Cumulative Singular Value Energy
        energy_ratio = np.sum(S[:k]) / total_energy

        metrics["rank"].append(k)
        metrics["frob_error"].append(float(err_fro))
        metrics["compression_ratio"].append(float(cr))
        metrics["energy_ratio"].append(float(energy_ratio))
        metrics["reconstructed_images"][k] = reconstructed

    # Plot 1: Visual comparison grid
    num_plots = len(ranks) + 1
    cols = 4
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    # Original
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original ({M}x{N})\n100% Params", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for idx, k in enumerate(ranks):
        ax = axes[idx + 1]
        recon = metrics["reconstructed_images"][k]
        cr = metrics["compression_ratio"][idx]
        err = metrics["frob_error"][idx]
        energy = metrics["energy_ratio"][idx]
        
        ax.imshow(np.clip(recon, 0.0, 1.0), cmap="gray")
        ax.set_title(
            f"Rank k={k} ({cr:.2f}x)\n"
            f"Frob Err: {err * 100:.2f}%\n"
            f"Energy: {energy * 100:.2f}%",
            fontsize=10
        )
        ax.axis("off")

    for j in range(num_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "svd_image_reconstruction_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()

    # Plot 2: Quantitative Spectral & Error Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Singular Value Spectrum Decay (Log scale)
    ax1.semilogy(range(1, min_dim + 1), S, color="#1f77b4", lw=2, label=r"Singular Values $\sigma_i$")
    for k in [8, 32, 64, 128]:
        ax1.axvline(k, color="crimson", linestyle="--", alpha=0.6, label=f"Rank {k}" if k == 8 else None)
    ax1.set_xlabel("Singular Value Index $i$", fontsize=11)
    ax1.set_ylabel(r"Singular Value Magnitude $\sigma_i$ (log scale)", fontsize=11)
    ax1.set_title("Singular Value Spectrum Decay", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    # Right: Tradeoff Pareto (Frobenius Error vs Compression Ratio)
    color_err = "#d62728"
    color_nrg = "#2ca02c"
    
    ax2.plot(metrics["rank"], [e * 100 for e in metrics["frob_error"]], "o-", color=color_err, label="Frobenius Error (%)")
    ax2.set_xlabel("Truncated Rank $k$", fontsize=11)
    ax2.set_ylabel("Frobenius Error (%)", color=color_err, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_err)
    ax2.grid(True, ls="--", alpha=0.4)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(metrics["rank"], [e * 100 for e in metrics["energy_ratio"]], "s--", color=color_nrg, label="Energy Retained (%)")
    ax2_twin.set_ylabel("Singular Value Energy (%)", color=color_nrg, fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor=color_nrg)

    ax2.set_title("Reconstruction Error & Energy vs. Rank", fontsize=12, fontweight="bold")

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "svd_image_metrics_tradeoff.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    print(f"[Phase 1 SVD] Plots saved to {grid_path} and {curves_path}")
    return metrics


if __name__ == "__main__":
    img_mat = generate_or_load_test_image(size=512)
    results = run_image_svd(img_mat)
    print("\n=== Phase 1 SVD Quantitative Summary ===")
    print(f"{'Rank':<8} | {'Compression':<14} | {'Frob Error (%)':<16} | {'Energy Retained (%)':<20}")
    print("-" * 65)
    for r, cr, err, nrg in zip(
        results["rank"],
        results["compression_ratio"],
        results["frob_error"],
        results["energy_ratio"]
    ):
        print(f"{r:<8} | {cr:>10.2f}x    | {err * 100:>13.2f}%   | {nrg * 100:>16.2f}%")
