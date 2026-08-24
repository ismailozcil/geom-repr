import torch


def apply_csdm(p_init: torch.Tensor, W_matr: torch.Tensor, gamma: float = 3.0) -> torch.Tensor:
    """Applies exponential Sinkhorn kernel matrix to initial ensemble probabilities[cite: 2].
    
    Args:
        p_init: Initial prediction probabilities of shape (N, C)[cite: 2].
        W_matr: Pairwise distance/cost matrix of shape (C, C)[cite: 2].
        gamma: Kernel scaling parameter[cite: 2].
        
    Returns:
        p_csdm: Smoothed scores of shape (N, C)[cite: 2].
    """
    S = torch.exp(-gamma * W_matr).float() #[cite: 2]
    return (S @ p_init.t()).t() #[cite: 2]
