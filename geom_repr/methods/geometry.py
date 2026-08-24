import math
from typing import Optional, Tuple
import torch


def zero_mean(m: torch.Tensor) -> torch.Tensor: #[cite: 3]
    """Column-wise mean-centering of a (D, N) matrix[cite: 3]."""
    return m - m.mean(dim=1, keepdim=True) #[cite: 2, 3]


def nearest_columns(m: torch.Tensor, query: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]: #[cite: 3]
    """Returns the k columns of m (D, N) closest to query (D, 1) in L2 distance[cite: 3]."""
    k = min(k, m.shape[1]) #[cite: 2, 3]
    d = torch.sum((m - query) ** 2, dim=0) #[cite: 3]
    idx = torch.argsort(d)[:k] #[cite: 3]
    return m[:, idx], idx #[cite: 3]


def principal_angle_score(
    neighborhood: torch.Tensor, query: torch.Tensor, weighted: bool = True, eps: float = 1e-12 #[cite: 3]
) -> torch.Tensor:
    """Calculates subspace curvature consistency score in [0, 1][cite: 3]."""
    n = neighborhood.shape[1] #[cite: 3]
    if n == 0: #[cite: 3]
        return torch.tensor(0.0) #[cite: 2, 3]

    neigh_c = zero_mean(neighborhood) #[cite: 2, 3]
    combined_c = zero_mean(torch.cat([neighborhood, query], dim=1)) #[cite: 2, 3]

    U_n, S_n, _ = torch.linalg.svd(neigh_c, full_matrices=True) #[cite: 2, 3]
    U_c, S_c, _ = torch.linalg.svd(combined_c, full_matrices=True) #[cite: 2, 3]

    r = min(n, U_n.shape[1], U_c.shape[1]) #[cite: 3]
    U_n, S_n = U_n[:, :r], S_n[:r] #[cite: 2, 3]
    U_c, S_c = U_c[:, :r], S_c[:r] #[cite: 2, 3]

    if weighted: #[cite: 3]
        A = (U_n * S_n).T @ (U_c * S_c) #[cite: 2, 3]
        denom = torch.sum(S_n * S_c) #[cite: 2, 3]
    else:
        A = U_n.T @ U_c #[cite: 3]
        denom = torch.tensor(float(r)) #[cite: 3]

    sv = torch.linalg.svdvals(A) #[cite: 3]
    numer = torch.sum(sv) #[cite: 2, 3]

    if denom <= eps or not torch.isfinite(denom) or not torch.isfinite(numer): #[cite: 2, 3]
        return torch.tensor(0.0) #[cite: 2, 3]

    ratio = torch.clamp(torch.abs(numer / denom), 0.0, 1.0) #[cite: 2, 3]
    theta = torch.acos(ratio) #[cite: 2, 3]
    return 1.0 - 2.0 * theta / math.pi #[cite: 2, 3]


def leverage_scores(m: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor: #[cite: 3]
    """Calculates statistical leverage scores for each column of m (D, N)[cite: 3]."""
    centered = zero_mean(m) #[cite: 3]
    U, S, _ = torch.linalg.svd(centered, full_matrices=False) #[cite: 3]
    r = min(rank or U.shape[1], U.shape[1]) #[cite: 3]
    Uk = U[:, :r] #[cite: 3]
    return torch.sum((centered.T @ Uk) ** 2 / (S[:r] ** 2 + 1e-12), dim=1) #[cite: 3]
