"""
Decomposed Linear Module & SVD Layer-Swapping Mechanics.
Person A / Co-developed:
Replaces W in R^{M x N} with two cascaded low-rank linear layers:
L1 (M -> K, bias=False) and L2 (K -> N, bias=True).

Supports:
- SVD factorization of nn.Linear weights: W = U * diag(S) * Vt
- Truncation to rank K: U_k in R^{M x K}, S_k in R^K, Vt_k in R^{K x N}
- Singular value distribution schemes:
  * "symmetric": fold sqrt(S_k) into both L1 and L2
  * "asymmetric_u": fold S_k into L1 (or L2)
- In-place model transformation: swaps nn.Linear instances with DecomposedLinear.
"""

import copy
import math
import torch
import torch.nn as nn


class DecomposedLinear(nn.Module):
    """
    Cascaded linear layers approximating W in R^{M x N} via rank K:
    y = (x @ L1^T) @ L2^T + bias
    where L1: M -> K (bias=False)
          L2: K -> N (bias=has_bias)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # L1 maps in_features -> rank (No bias)
        self.L1 = nn.Linear(in_features, rank, bias=False)
        # L2 maps rank -> out_features (Optional bias)
        self.L2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.L1(x)
        out = self.L2(h)
        return out

    @classmethod
    def from_linear(
        cls,
        linear_module: nn.Linear,
        rank: int,
        svd_scheme: str = "symmetric"
    ) -> "DecomposedLinear":
        """
        Constructs a DecomposedLinear by performing truncated SVD on an existing nn.Linear.
        PyTorch nn.Linear stores weight in shape (out_features, in_features).
        Forward pass is x @ W^T + b.
        Let W_mat = linear_module.weight.data (shape [N_out, M_in]).
        SVD on W_mat: W_mat = U * diag(S) * Vt
        where:
          U is (N_out, min(N, M))
          S is (min(N, M),)
          Vt is (min(N, M), M_in)
        
        Truncated to rank K:
          U_k: (N_out, K)
          S_k: (K,)
          Vt_k: (K, M_in)
          
        If symmetric:
          L1_weight = diag(sqrt(S_k)) @ Vt_k -> shape (K, M_in)
          L2_weight = U_k @ diag(sqrt(S_k))  -> shape (N_out, K)
        """
        has_bias = linear_module.bias is not None
        in_f = linear_module.in_features
        out_f = linear_module.out_features
        k = min(rank, in_f, out_f)

        decomposed = cls(in_f, out_f, k, bias=has_bias)

        # PyTorch weight shape: [out_features, in_features]
        # We perform SVD on float32/float64 for numerical precision
        W = linear_module.weight.detach().to(torch.float32)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        U_k = U[:, :k]       # [out_features, k]
        S_k = S[:k]         # [k]
        Vt_k = Vt[:k, :]    # [k, in_features]

        if svd_scheme == "symmetric":
            sqrt_S = torch.diag(torch.sqrt(S_k))
            # L1 maps in_features -> k, PyTorch weight shape is [k, in_features]
            w_l1 = sqrt_S @ Vt_k
            # L2 maps k -> out_features, PyTorch weight shape is [out_features, k]
            w_l2 = U_k @ sqrt_S
        elif svd_scheme == "fold_u":
            w_l1 = Vt_k
            w_l2 = U_k @ torch.diag(S_k)
        elif svd_scheme == "fold_v":
            w_l1 = torch.diag(S_k) @ Vt_k
            w_l2 = U_k
        else:
            raise ValueError(f"Unknown SVD scheme: {svd_scheme}")

        decomposed.L1.weight.data.copy_(w_l1.to(linear_module.weight.dtype))
        decomposed.L2.weight.data.copy_(w_l2.to(linear_module.weight.dtype))

        if has_bias:
            decomposed.L2.bias.data.copy_(linear_module.bias.detach())

        return decomposed


def decompose_model_linear_layers(
    model: nn.Module,
    rank: int,
    target_layers: list[str] | None = None,
    svd_scheme: str = "symmetric"
) -> nn.Module:
    """
    Returns a cloned model where designated (or all wide) nn.Linear layers
    are replaced with DecomposedLinear modules.
    """
    decomposed_model = copy.deepcopy(model)

    for name, module in list(decomposed_model.named_modules()):
        if isinstance(module, nn.Linear):
            if target_layers is not None and name not in target_layers:
                continue
            
            # Don't decompose final small classifier layer (e.g. 10 classes) if rank >= classes
            if module.out_features <= 10 or module.in_features <= 10:
                continue

            k = min(rank, module.in_features, module.out_features)
            decomposed_layer = DecomposedLinear.from_linear(module, k, svd_scheme=svd_scheme)
            
            # Replace layer in parent module
            parent_module = decomposed_model
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], decomposed_layer)

    return decomposed_model
