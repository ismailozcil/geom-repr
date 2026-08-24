import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from sklearn.metrics import average_precision_score #[cite: 3]
from geom_repr.methods.geometry import (
    zero_mean,
    nearest_columns,
    principal_angle_score,
    leverage_scores,
)


@dataclass
class PrototypeBank:
    class_features: Dict[int, torch.Tensor] = field(default_factory=dict) #[cite: 3]
    class_names: Dict[int, str] = field(default_factory=dict) #[cite: 3]
    spm_proj: Dict[int, torch.Tensor] = field(default_factory=dict) #[cite: 3]
    spm_mean: Dict[int, torch.Tensor] = field(default_factory=dict) #[cite: 3]
    spm_threshold: Dict[int, float] = field(default_factory=dict) #[cite: 3]
    mcm_threshold: Dict[int, float] = field(default_factory=dict) #[cite: 3]

    # Discriminative SPM (SPM-D) caches[cite: 3]
    spm_disc_dirs: Dict[int, torch.Tensor] = field(default_factory=dict) #[cite: 3]
    spm_disc_signs: Dict[int, torch.Tensor] = field(default_factory=dict) #[cite: 3]
    spm_disc_mean: Dict[int, torch.Tensor] = field(default_factory=dict) #[cite: 3]
    spm_disc_threshold: Dict[int, float] = field(default_factory=dict) #[cite: 3]

    @classmethod
    def build(
        cls, features: torch.Tensor, labels: torch.Tensor, class_names: Optional[Sequence[str]] = None #[cite: 3]
    ) -> "PrototypeBank":
        """Builds class-wise prototype representations from feature matrix (N, D) and labels (N, C)[cite: 3]."""
        bank = cls() #[cite: 3]
        feats_t = features.T  # Shape: (D, N)[cite: 3]
        n_classes = labels.shape[1] #[cite: 3]
        for c in range(n_classes): #[cite: 3]
            mask = labels[:, c].bool() #[cite: 3]
            if mask.sum() == 0: #[cite: 3]
                continue #[cite: 3]
            bank.class_features[c] = feats_t[:, mask].clone() #[cite: 3]
            bank.class_names[c] = class_names[c] if class_names else str(c) #[cite: 3]
        return bank #[cite: 3]

    @property
    def class_ids(self) -> List[int]:
        return list(self.class_features.keys()) #[cite: 3]

    # ------------------ SPM ------------------
    def fit_spm(self, rank: int = 8) -> None: #[cite: 3]
        """Fits standard PCA projection matrices for each class subspace[cite: 2, 3]."""
        for c, feats in self.class_features.items(): #[cite: 3]
            mean = feats.mean(dim=1, keepdim=True) #[cite: 2, 3]
            centered = feats - mean #[cite: 2, 3]
            U, _, _ = torch.linalg.svd(centered, full_matrices=False) #[cite: 2, 3]
            d = min(rank, U.shape[1]) #[cite: 3]
            Ud = U[:, :d] #[cite: 3]
            self.spm_proj[c] = Ud @ Ud.T #[cite: 3]
            self.spm_mean[c] = mean #[cite: 3]

    def spm_score(self, query: torch.Tensor, c: int) -> float: #[cite: 3]
        """Calculates ratio-of-norms projection alignment[cite: 2, 3]."""
        centered = query - self.spm_mean[c].squeeze() #[cite: 3]
        denom = torch.norm(centered) #[cite: 2, 3]
        if denom < 1e-12: #[cite: 3]
            return 0.0 #[cite: 3]
        num = torch.norm(self.spm_proj[c] @ centered) #[cite: 2, 3]
        return (num / denom).item() #[cite: 2, 3]

    def calibrate_spm_thresholds(
        self, features: torch.Tensor, labels: torch.Tensor, n_steps: int = 100 #[cite: 3]
    ) -> None:
        """Finds ROC-corner optimal thresholds for standard SPM[cite: 2, 3]."""
        for c in self.class_ids: #[cite: 3]
            scores = torch.tensor([self.spm_score(features[i], c) for i in range(len(features))]) #[cite: 3]
            pos = labels[:, c].bool() #[cite: 3]
            best_thresh, best_ts = 0.5, float("inf") #[cite: 3]
            for t in np.linspace(0, 1, n_steps): #[cite: 3]
                pred = scores > t #[cite: 3]
                tp = (pred & pos).sum().item() #[cite: 2, 3]
                fn = (~pred & pos).sum().item() #[cite: 2, 3]
                fp = (pred & ~pos).sum().item() #[cite: 2, 3]
                tn = (~pred & ~pos).sum().item() #[cite: 2, 3]
                tpr = tp / (tp + fn) if (tp + fn) else 0.0 #[cite: 2, 3]
                fpr = fp / (fp + tn) if (fp + tn) else 0.0 #[cite: 2, 3]
                ts = math.sqrt((1 - tpr) ** 2 + fpr ** 2) #[cite: 2, 3]
                if ts < best_ts: #[cite: 2, 3]
                    best_ts, best_thresh = ts, t #[cite: 2, 3]
            self.spm_threshold[c] = best_thresh #[cite: 3]

    # ------------------ MCM ------------------
    def mcm_score(
        self, query: torch.Tensor, c: int, k: int, feats_override: Optional[torch.Tensor] = None #[cite: 3]
    ) -> float:
        """Calculates manifold curvature consistency against k-nearest neighbors[cite: 2, 3]."""
        feats = self.class_features[c] if feats_override is None else feats_override #[cite: 3]
        neigh, _ = nearest_columns(feats, query.unsqueeze(1), k) #[cite: 3]
        return principal_angle_score(neigh, query.unsqueeze(1)).item() #[cite: 3]

    def calibrate_mcm_thresholds(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        k_neighbors: int,
        n_steps: int = 100, #[cite: 3]
        max_negatives: Optional[int] = 300, #[cite: 3]
        seed: int = 0, #[cite: 3]
    ) -> None:
        """Finds ROC-corner optimal thresholds for MCM with leave-one-out positives[cite: 3]."""
        rng = np.random.default_rng(seed) #[cite: 3]
        for c in self.class_ids: #[cite: 3]
            feats = self.class_features[c] #[cite: 3]
            n_c = feats.shape[1] #[cite: 3]

            pos_scores = [] #[cite: 3]
            for j in range(n_c): #[cite: 3]
                mask = torch.ones(n_c, dtype=torch.bool) #[cite: 3]
                mask[j] = False #[cite: 3]
                pool = feats[:, mask] #[cite: 3]
                query = feats[:, j] #[cite: 3]
                pos_scores.append(self.mcm_score(query, c, k_neighbors, feats_override=pool)) #[cite: 3]
            pos_scores = torch.tensor(pos_scores) #[cite: 3]

            neg_idx_all = torch.nonzero(~labels[:, c].bool(), as_tuple=True)[0] #[cite: 3]
            if max_negatives is not None and len(neg_idx_all) > max_negatives: #[cite: 3]
                sel = rng.choice(len(neg_idx_all), size=max_negatives, replace=False) #[cite: 3]
                neg_idx = neg_idx_all[sel] #[cite: 3]
            else:
                neg_idx = neg_idx_all #[cite: 3]

            neg_scores = torch.tensor([self.mcm_score(features[i], c, k_neighbors) for i in neg_idx.tolist()]) #[cite: 3]

            best_thresh, best_ts = 0.5, float("inf") #[cite: 3]
            for t in np.linspace(0, 1, n_steps): #[cite: 3]
                tp = (pos_scores > t).sum().item() #[cite: 3]
                fn = (pos_scores <= t).sum().item() #[cite: 3]
                fp = (neg_scores > t).sum().item() #[cite: 3]
                tn = (neg_scores <= t).sum().item() #[cite: 3]
                tpr = tp / (tp + fn) if (tp + fn) else 0.0 #[cite: 3]
                fpr = fp / (fp + tn) if (fp + tn) else 0.0 #[cite: 3]
                ts = math.sqrt((1 - tpr) ** 2 + fpr ** 2) #[cite: 3]
                if ts < best_ts: #[cite: 3]
                    best_ts, best_thresh = ts, t #[cite: 3]
            self.mcm_threshold[c] = best_thresh #[cite: 3]

    # ------------------ SPM-Discriminative (SPM-D) ------------------
    def fit_spm_discriminative(
        self, features: torch.Tensor, labels: torch.Tensor, rank: int = 8, candidate_rank: Optional[int] = None #[cite: 3]
    ) -> None:
        """Finds discriminative singular directions weighted by Average Precision[cite: 3]."""
        for c, feats in self.class_features.items(): #[cite: 3]
            mean = feats.mean(dim=1, keepdim=True) #[cite: 3]
            centered = feats - mean #[cite: 3]
            U, S, _ = torch.linalg.svd(centered, full_matrices=False) #[cite: 3]
            k_cand = U.shape[1] if candidate_rank is None else min(candidate_rank, U.shape[1]) #[cite: 3]
            U_cand = U[:, :k_cand] #[cite: 3]

            all_centered = features - mean.squeeze(1) #[cite: 3]
            projections_all = (all_centered @ U_cand).numpy() #[cite: 3]
            pos_np = labels[:, c].numpy() #[cite: 3]

            disc_scores, signs = [], [] #[cite: 3]
            for k in range(k_cand): #[cite: 3]
                s = projections_all[:, k] #[cite: 3]
                ap1 = average_precision_score(pos_np, s) #[cite: 3]
                ap2 = average_precision_score(pos_np, -s) #[cite: 3]
                if ap1 >= ap2: #[cite: 3]
                    disc_scores.append(ap1) #[cite: 3]
                    signs.append(1.0) #[cite: 3]
                else:
                    disc_scores.append(ap2) #[cite: 3]
                    signs.append(-1.0) #[cite: 3]

            disc_scores = np.array(disc_scores) #[cite: 3]
            d = min(rank, k_cand) #[cite: 3]
            top_idx = np.argsort(-disc_scores)[:d] #[cite: 3]

            self.spm_disc_dirs[c] = U_cand[:, top_idx] #[cite: 3]
            self.spm_disc_signs[c] = torch.tensor(np.array(signs)[top_idx], dtype=torch.float32) #[cite: 3]
            self.spm_disc_mean[c] = mean #[cite: 3]

    def spm_disc_score(self, query: torch.Tensor, c: int) -> float: #[cite: 3]
        """Evaluates query using signed sum of discriminative projections[cite: 3]."""
        centered = query - self.spm_disc_mean[c].squeeze(1) #[cite: 3]
        proj = self.spm_disc_dirs[c].T @ centered #[cite: 3]
        return (proj * self.spm_disc_signs[c]).sum().item() #[cite: 3]

    def calibrate_spm_disc_thresholds(
        self, features: torch.Tensor, labels: torch.Tensor, n_steps: int = 100 #[cite: 3]
    ) -> None:
        """Finds ROC-corner optimal thresholds for SPM-D[cite: 3]."""
        for c in self.class_ids: #[cite: 3]
            scores = torch.tensor([self.spm_disc_score(features[i], c) for i in range(len(features))]) #[cite: 3]
            pos = labels[:, c].bool() #[cite: 3]
            lo, hi = scores.min().item(), scores.max().item() #[cite: 3]
            best_thresh, best_ts = 0.0, float("inf") #[cite: 3]
            for t in np.linspace(lo, hi, n_steps): #[cite: 3]
                pred = scores > t #[cite: 3]
                tp = (pred & pos).sum().item() #[cite: 3]
                fn = (~pred & pos).sum().item() #[cite: 3]
                fp = (pred & ~pos).sum().item() #[cite: 3]
                tn = (~pred & ~pos).sum().item() #[cite: 3]
                tpr = tp / (tp + fn) if (tp + fn) else 0.0 #[cite: 3]
                fpr = fp / (fp + tn) if (fp + tn) else 0.0 #[cite: 3]
                ts = math.sqrt((1 - tpr) ** 2 + fpr ** 2) #[cite: 3]
                if ts < best_ts: #[cite: 3]
                    best_ts, best_thresh = ts, t #[cite: 3]
            self.spm_disc_threshold[c] = best_thresh #[cite: 3]

    # ------------------ Pruning ------------------
    def prune_class(
        self,
        c: int,
        k_neighbors: int = 5,
        angle_threshold: float = 0.02,
        mode: str = "sequential",
        rng_seed: int = 0, #[cite: 3]
    ) -> torch.Tensor:
        """Prunes redundant prototypes using leverage scoring or curvature consistency[cite: 3]."""
        feats = self.class_features[c] #[cite: 3]
        n_total = feats.shape[1] #[cite: 3]

        if mode == "leverage": #[cite: 3]
            scores = leverage_scores(feats) #[cite: 3]
            keep_n = max(k_neighbors + 1, int(0.5 * n_total)) #[cite: 3]
            keep_idx = torch.argsort(scores, descending=True)[:keep_n] #[cite: 3]
            return feats[:, torch.sort(keep_idx).values] #[cite: 3]

        if mode != "sequential": #[cite: 3]
            raise ValueError("mode must be 'sequential' or 'leverage'") #[cite: 3]

        g = torch.Generator().manual_seed(rng_seed) #[cite: 3]
        order = torch.randperm(n_total, generator=g) #[cite: 3]

        keep_mask = torch.ones(n_total, dtype=torch.bool) #[cite: 3]
        for idx in order.tolist(): #[cite: 3]
            kept_idx = torch.nonzero(keep_mask, as_tuple=True)[0] #[cite: 3]
            kept_idx = kept_idx[kept_idx != idx] #[cite: 3]
            if kept_idx.numel() < 2: #[cite: 3]
                continue #[cite: 3]
            candidate_pool = feats[:, kept_idx] #[cite: 3]
            query = feats[:, idx].unsqueeze(1) #[cite: 3]
            neigh, _ = nearest_columns(candidate_pool, query, k_neighbors) #[cite: 3]
            consistency = principal_angle_score(neigh, query).item() #[cite: 3]
            if consistency >= (1.0 - angle_threshold): #[cite: 3]
                keep_mask[idx] = False #[cite: 3]

        kept_idx = torch.nonzero(keep_mask, as_tuple=True)[0] #[cite: 3]
        return feats[:, kept_idx] #[cite: 3]
