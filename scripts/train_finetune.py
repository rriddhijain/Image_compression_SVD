"""
Baseline Training and SVD Fine-Tuning Pipeline.
Person A Role Support / Full-Stack Reproducibility.

1. Trains BaselineMLP until convergence (~86-88% accuracy).
2. Saves results/checkpoints/baseline_model.pt.
3. Evaluates zero-shot accuracy post-SVD decomposition across ranks.
4. Performs fine-tuning (Sainath et al., 2013) with AdamW, cosine annealing (5-10 epochs).
5. Saves results/checkpoints/svd_k{rank}_finetuned.pt and logs Pareto metrics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline import BaselineMLP
from models.decomposed import decompose_model_linear_layers
from profiling.metrics import model_theoretical_summary
from visualizations.pareto import plot_pareto_compression_vs_accuracy
from data.dataset_loader import get_benchmark_dataloaders


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def train_baseline(
    epochs: int = 5,
    device_name: str | None = None
) -> tuple[BaselineMLP, float]:
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    train_loader, test_loader, in_dim = get_benchmark_dataloaders()

    model = BaselineMLP(in_features=in_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Training Baseline] Model on {device} ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {running_loss / len(train_loader):.4f} | Test Acc: {acc * 100:.2f}%")

    os.makedirs("results/checkpoints", exist_ok=True)
    save_path = "results/checkpoints/baseline_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[Baseline Saved] {save_path} with final test accuracy: {acc * 100:.2f}%")
    return model, acc


def run_svd_finetuning_pipeline(
    ranks: list[int] = [8, 16, 32, 64, 128, 256],
    finetune_epochs: int = 3,
    device_name: str | None = None
):
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    train_loader, test_loader, in_dim = get_benchmark_dataloaders()

    checkpoint_path = "results/checkpoints/baseline_model.pt"
    if not os.path.exists(checkpoint_path):
        base_model, baseline_acc = train_baseline(epochs=4, device_name=str(device))
    else:
        base_model = BaselineMLP(in_features=in_dim, num_classes=10)
        base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        base_model.to(device)
        baseline_acc = evaluate(base_model, test_loader, device)

    base_params = sum(p.numel() for p in base_model.parameters())
    records = [{
        "rank": None,
        "is_decomposed": False,
        "compression_ratio": 1.0,
        "param_count": base_params,
        "zero_shot_acc": baseline_acc,
        "finetuned_acc": baseline_acc,
        "accuracy": baseline_acc
    }]

    print(f"\nBaseline Model Parameters: {base_params:,} | Accuracy: {baseline_acc * 100:.2f}%\n")
    print(f"{'Rank':<6} | {'Params':<10} | {'Comp. Ratio':<12} | {'Zero-Shot Acc':<15} | {'Finetuned Acc':<15}")
    print("-" * 70)

    for k in ranks:
        # Step 1: Decompose layer via SVD
        decomp_model = decompose_model_linear_layers(base_model, rank=k).to(device)
        decomp_params = sum(p.numel() for p in decomp_model.parameters())
        cr = base_params / decomp_params

        # Step 2: Measure Zero-Shot Accuracy
        zero_shot_acc = evaluate(decomp_model, test_loader, device)

        # Step 3: Fine-tune (Sainath et al., 2013)
        optimizer = torch.optim.AdamW(decomp_model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        criterion = nn.CrossEntropyLoss()

        for ep in range(finetune_epochs):
            decomp_model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = decomp_model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        finetuned_acc = evaluate(decomp_model, test_loader, device)

        # Save checkpoint
        ckpt_out = f"results/checkpoints/svd_k{k}_finetuned.pt"
        torch.save(decomp_model.state_dict(), ckpt_out)

        records.append({
            "rank": k,
            "is_decomposed": True,
            "compression_ratio": cr,
            "param_count": decomp_params,
            "zero_shot_acc": zero_shot_acc,
            "finetuned_acc": finetuned_acc,
            "accuracy": finetuned_acc
        })

        print(f"k={k:<4} | {decomp_params:<10,} | {cr:>10.2f}x | {zero_shot_acc * 100:>13.2f}% | {finetuned_acc * 100:>13.2f}%")

    df_acc = pd.DataFrame(records)
    df_acc.to_csv("results/accuracy_pareto_data.csv", index=False)
    plot_pareto_compression_vs_accuracy(df_acc, output_path="results/plots/pareto_compression_vs_accuracy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run_svd_finetuning_pipeline(finetune_epochs=args.epochs)
