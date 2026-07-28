"""Chromosome-chunked dataset for REFCON.

Key features:
  - Within-chromosome chunks only (CN does not cross chromosome boundaries)
  - Pad small chromosomes, sub-chunk large ones
  - Three normalization modes: "raw" (log1p), "cpm" (CPM+log1p), "dual" (both)
  - Full data coverage: __len__ = n_cells * n_chromosomes
  - Loss-region oversampling via WeightedRandomSampler
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# CN thresholds
CN_LOSS_THRESH = 1.5
CN_GAIN_THRESH = 2.5
CN_IGNORE = -1
TARGET_SUM = 10000.0
MAX_CHUNK = 500


def _categorize_cn(cn_values: np.ndarray) -> np.ndarray:
    """Convert raw CN float values to integer labels (loss=0, neutral=1, gain=2, NaN->-1)."""
    labels = np.full(len(cn_values), 1, dtype=np.int8)
    labels[cn_values < CN_LOSS_THRESH] = 0
    labels[cn_values > CN_GAIN_THRESH] = 2
    labels[np.isnan(cn_values)] = CN_IGNORE
    return labels


# ======================================================================
# Chromosome-Chunked Dataset
# ======================================================================

class ChrDataset(Dataset):
    """Within-chromosome chunked dataset.

    Two coverage modes:
      - "full": __len__ = n_cells * n_chromosomes. Every (cell, chr) seen per epoch.
      - "random": __len__ = n_cells. Each cell picks a random chromosome per call.

    Args:
        h5ad_path: Path to a processed h5ad file.
        cn_type: CN label type (bulk, per_line, per_cell, pseudobulk). Defaults to the
                 value stored in ``adata.uns["cn_type"]``.
        norm_mode: "raw" (log1p), "cpm" (CPM+log1p), or "dual" (both raw + cpm).
        max_chunk: Maximum chunk length. Chromosomes > max_chunk are sub-chunked.
        coverage: "full" (n_cells * n_chrs samples) or "random" (n_cells samples, random chr).
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        cn_type: str | None = None,
        norm_mode: str = "raw",
        max_chunk: int = MAX_CHUNK,
        coverage: str = "full",
    ):
        self.max_chunk = max_chunk
        self.norm_mode = norm_mode
        self.coverage = coverage

        adata = ad.read_h5ad(str(h5ad_path))
        self.dataset_name = adata.uns.get("dataset_name", Path(h5ad_path).stem)
        if cn_type is None:
            cn_type = adata.uns.get("cn_type", "per_cell")
        if adata.n_obs == 0:
            raise ValueError(f"No cells in {h5ad_path}")

        cn_profiles_dict = adata.uns.get("cn_profiles", {})
        cn_matrix_full = adata.uns.get("cn_matrix", None)
        cn_cell_ids_full = adata.uns.get("cn_cell_ids", None)

        cell_indices = np.arange(adata.n_obs)
        adata_sub = adata

        self.n_cells = adata_sub.n_obs
        self.n_genes = adata_sub.n_vars

        # Expression matrix (keep sparse)
        self.X = adata_sub.X.copy() if sp.issparse(adata_sub.X) else adata_sub.X.copy()

        # Chromosome boundaries
        self.chr_boundary = adata_sub.var["chr_boundary"].values.astype(np.float32).copy()
        self.chr_ranges = self._build_chromosome_ranges()
        self.n_chromosomes = len(self.chr_ranges)

        # Build CN labels
        self._build_cn(cn_type, adata_sub, cn_profiles_dict, cn_matrix_full,
                        cn_cell_ids_full, cell_indices)

        # Precompute which chromosomes have loss genes (for oversampling)
        self._loss_chr_mask = self._build_loss_chr_mask()

        logger.info(
            "%s loaded: %d cells x %d genes x %d chrs = %d samples, norm=%s, coverage=%s",
            self.dataset_name, self.n_cells, self.n_genes,
            self.n_chromosomes, len(self), self.norm_mode, self.coverage,
        )

    # ------------------------------------------------------------------
    # Chromosome ranges
    # ------------------------------------------------------------------

    def _build_chromosome_ranges(self) -> list[tuple[int, int]]:
        """Precompute (start, end) for each chromosome from chr_boundary."""
        starts = np.where(self.chr_boundary > 0.5)[0]
        ranges = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else self.n_genes
            ranges.append((s, e))
        return ranges

    def _build_loss_chr_mask(self) -> np.ndarray:
        """Boolean mask: which chromosomes contain loss genes.

        Returns (n_chromosomes,) bool array.
        """
        if self._cn_mode == "shared":
            cn = self._cn_continuous_shared
        elif self._cn_mode == "per_line":
            # Union across all profiles
            cn = self._cn_continuous_profiles.min(axis=0)
        else:
            cn = self._cn_continuous_matrix.min(axis=0)

        mask = np.zeros(self.n_chromosomes, dtype=bool)
        for ci, (s, e) in enumerate(self.chr_ranges):
            chr_cn = cn[s:e]
            valid = ~np.isnan(chr_cn) & (chr_cn >= 0)
            if valid.any() and (chr_cn[valid] < CN_LOSS_THRESH).any():
                mask[ci] = True
        return mask

    # ------------------------------------------------------------------
    # CN labels
    # ------------------------------------------------------------------

    def _build_cn(self, cn_type, adata_sub, cn_profiles_dict, cn_matrix_full,
                  cn_cell_ids_full, cell_indices):
        if cn_type in ("bulk", "pseudobulk"):
            self._cn_continuous_shared = adata_sub.var["cn_ground_truth"].values.astype(
                np.float32
            ).copy()
            self._cn_mode = "shared"

        elif cn_type == "per_line":
            profile_names = adata_sub.obs["cn_profile"].values
            unique_names = sorted(set(profile_names))
            name_to_idx = {n: i for i, n in enumerate(unique_names)}

            profiles_continuous = []
            for name in unique_names:
                raw = np.array(cn_profiles_dict[name], dtype=np.float32)
                profiles_continuous.append(raw.copy())
            self._cn_continuous_profiles = np.stack(profiles_continuous)
            self._cn_cell_idx = np.array(
                [name_to_idx[n] for n in profile_names], dtype=np.int32
            )
            self._cn_mode = "per_line"

        elif cn_type == "per_cell":
            if cn_matrix_full is not None and cn_cell_ids_full is not None:
                id_to_row = {cid: i for i, cid in enumerate(cn_cell_ids_full)}
                sub_ids = adata_sub.obs.index.values
                rows = np.array([id_to_row[cid] for cid in sub_ids], dtype=np.int64)
                cn_sub = cn_matrix_full[rows]
            elif cn_matrix_full is not None:
                cn_sub = cn_matrix_full[cell_indices]
            else:
                raise ValueError("per_cell cn_type but no cn_matrix in uns")
            self._cn_continuous_matrix = cn_sub.astype(np.float32).copy()
            self._cn_mode = "per_cell"
        else:
            raise ValueError(f"Unknown cn_type: {cn_type}")

    def _get_cn_continuous(self, idx: int) -> np.ndarray:
        """(n_genes,) float32 continuous CN values. NO capping."""
        if self._cn_mode == "shared":
            return self._cn_continuous_shared
        elif self._cn_mode == "per_line":
            return self._cn_continuous_profiles[self._cn_cell_idx[idx]]
        else:
            return self._cn_continuous_matrix[idx]

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize raw counts based on norm_mode.

        For "dual" mode, returns raw log1p (use _normalize_dual for both).
        """
        if self.norm_mode == "cpm":
            total = x.sum()
            if total > 0:
                x = x * (TARGET_SUM / total)
            return np.log1p(x)
        else:
            # "raw" or "dual": just log1p. InstanceNorm in model handles the rest.
            return np.log1p(x)

    def _normalize_dual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute both raw log1p and CPM+log1p from raw counts.

        Must be called on the full cell before chunking (CPM needs total count).
        """
        x_raw = np.log1p(x)
        x_cpm = x.copy()
        total = x_cpm.sum()
        if total > 0:
            x_cpm = x_cpm * (TARGET_SUM / total)
        x_cpm = np.log1p(x_cpm)
        return x_raw, x_cpm

    # ------------------------------------------------------------------
    # Full cell access (for evaluation)
    # ------------------------------------------------------------------

    def get_full_cell(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full-length (n_genes,) expression and CN for evaluation."""
        row = self.X[idx]
        if sp.issparse(row):
            x = row.toarray().ravel()
        else:
            x = np.asarray(row).ravel()
        expr = self._normalize(x.astype(np.float32))
        cn = self._get_cn_continuous(idx).astype(np.float32)
        cn = np.where(np.isnan(cn), -1.0, cn)
        return torch.from_numpy(expr), torch.from_numpy(cn)

    def get_full_cell_dual(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (expr_raw, expr_cpm, cn) full-length tensors for joint evaluation."""
        row = self.X[idx]
        if sp.issparse(row):
            x = row.toarray().ravel()
        else:
            x = np.asarray(row).ravel()
        x = x.astype(np.float32)
        expr_raw, expr_cpm = self._normalize_dual(x)
        cn = self._get_cn_continuous(idx).astype(np.float32)
        cn = np.where(np.isnan(cn), -1.0, cn)
        return torch.from_numpy(expr_raw), torch.from_numpy(expr_cpm), torch.from_numpy(cn)

    # ------------------------------------------------------------------
    # __len__ and __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.coverage == "full":
            return self.n_cells * self.n_chromosomes
        else:
            return self.n_cells

    def __getitem__(self, idx: int):
        """Returns sample tuple padded to max_chunk.

        For norm_mode="dual":
            (expr_raw, expr_cpm, cn, length) - 4-tuple
        Otherwise:
            (expression, cn, length) - 3-tuple
        """
        if self.coverage == "full":
            cell_idx = idx // self.n_chromosomes
            chr_idx = idx % self.n_chromosomes
        else:
            cell_idx = idx
            if hasattr(self, '_chr_sample_probs'):
                chr_idx = np.random.choice(self.n_chromosomes, p=self._chr_sample_probs)
            else:
                chr_idx = np.random.randint(0, self.n_chromosomes)
        chr_start, chr_end = self.chr_ranges[chr_idx]
        chr_size = chr_end - chr_start

        # Get raw expression for this cell
        row = self.X[cell_idx]
        if sp.issparse(row):
            x = row.toarray().ravel()
        else:
            x = np.asarray(row).ravel()
        x = x.astype(np.float32)

        # Normalize (full cell first, then chunk - important for CPM)
        if self.norm_mode == "dual":
            expr_raw_full, expr_cpm_full = self._normalize_dual(x)
        else:
            expr_full = self._normalize(x)

        # Get CN for this cell
        cn_full = self._get_cn_continuous(cell_idx).astype(np.float32)
        cn_full = np.where(np.isnan(cn_full), -1.0, cn_full)

        # Extract chromosome region
        if chr_size > self.max_chunk:
            # Random sub-chunk within chromosome
            max_start = chr_size - self.max_chunk
            offset = np.random.randint(0, max_start + 1)
            start = chr_start + offset
            end = start + self.max_chunk
            length = self.max_chunk
        else:
            start = chr_start
            end = chr_end
            length = chr_size

        # Build output chunks with padding
        cn_chunk = np.full(self.max_chunk, -1.0, dtype=np.float32)
        cn_chunk[:length] = cn_full[start:end]

        if self.norm_mode == "dual":
            raw_chunk = np.zeros(self.max_chunk, dtype=np.float32)
            cpm_chunk = np.zeros(self.max_chunk, dtype=np.float32)
            raw_chunk[:length] = expr_raw_full[start:end]
            cpm_chunk[:length] = expr_cpm_full[start:end]
            return (
                torch.from_numpy(raw_chunk),
                torch.from_numpy(cpm_chunk),
                torch.from_numpy(cn_chunk),
                length,
            )
        else:
            expr_chunk = np.zeros(self.max_chunk, dtype=np.float32)
            expr_chunk[:length] = expr_full[start:end]
            return (
                torch.from_numpy(expr_chunk),
                torch.from_numpy(cn_chunk),
                length,
            )

    # ------------------------------------------------------------------
    # Oversampling support
    # ------------------------------------------------------------------

    def get_sample_weights(self, loss_oversample: float = 2.0) -> np.ndarray | None:
        """Per-sample weights for WeightedRandomSampler.

        Only works for coverage="full". For "random", chromosome selection
        is randomized per call - use oversample_chr_probs instead.

        Args:
            loss_oversample: Weight multiplier for loss-containing chromosomes.

        Returns:
            (len(self),) float array of weights, or None for "random" mode.
        """
        if self.coverage == "random":
            # For random mode, build chromosome sampling probabilities
            # that bias toward loss chromosomes
            probs = np.ones(self.n_chromosomes, dtype=np.float64)
            probs[self._loss_chr_mask] *= loss_oversample
            self._chr_sample_probs = probs / probs.sum()
            return None

        weights = np.ones(len(self), dtype=np.float64)
        for chr_idx in range(self.n_chromosomes):
            if self._loss_chr_mask[chr_idx]:
                indices = np.arange(chr_idx, len(self), self.n_chromosomes)
                weights[indices] *= loss_oversample
        return weights


# ======================================================================
# Regression metrics
# ======================================================================

def compute_regression_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute regression + derived classification metrics from (n_cells, n_genes) tensors."""
    valid = labels > -0.5
    p = preds[valid].float()
    l = labels[valid].float()

    metrics: dict = {}
    if len(p) == 0:
        return {"mae": float("inf"), "rmse": float("inf")}

    diff = p - l
    metrics["mae"] = diff.abs().mean().item()
    metrics["rmse"] = diff.pow(2).mean().sqrt().item()

    ss_res = diff.pow(2).sum()
    ss_tot = (l - l.mean()).pow(2).sum()
    metrics["r2"] = (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()

    p_mean, l_mean = p.mean(), l.mean()
    cov = ((p - p_mean) * (l - l_mean)).mean()
    p_std = p.std().clamp(min=1e-8)
    l_std = l.std().clamp(min=1e-8)
    metrics["pearson"] = (cov / (p_std * l_std)).item()

    if len(p) > 10:
        p_rank = p.argsort().argsort().float()
        l_rank = l.argsort().argsort().float()
        pr_mean, lr_mean = p_rank.mean(), l_rank.mean()
        r_cov = ((p_rank - pr_mean) * (l_rank - lr_mean)).mean()
        pr_std = p_rank.std().clamp(min=1e-8)
        lr_std = l_rank.std().clamp(min=1e-8)
        metrics["spearman"] = (r_cov / (pr_std * lr_std)).item()
    else:
        metrics["spearman"] = 0.0

    for bin_name, lo, hi in [("loss", -0.5, CN_LOSS_THRESH),
                              ("neutral", CN_LOSS_THRESH, CN_GAIN_THRESH),
                              ("gain", CN_GAIN_THRESH, 100.0)]:
        mask = (l >= lo) & (l < hi)
        if mask.any():
            metrics[f"{bin_name}_mae"] = (p[mask] - l[mask]).abs().mean().item()
            metrics[f"{bin_name}_n"] = int(mask.sum())

    pred_cls = torch.ones_like(p, dtype=torch.long)
    pred_cls[p < CN_LOSS_THRESH] = 0
    pred_cls[p > CN_GAIN_THRESH] = 2
    true_cls = torch.ones_like(l, dtype=torch.long)
    true_cls[l < CN_LOSS_THRESH] = 0
    true_cls[l > CN_GAIN_THRESH] = 2

    metrics["derived_accuracy"] = (pred_cls == true_cls).float().mean().item()

    f1s = []
    for cls, name in enumerate(["loss", "neutral", "gain"]):
        tp = ((pred_cls == cls) & (true_cls == cls)).sum().float()
        fp = ((pred_cls == cls) & (true_cls != cls)).sum().float()
        fn = ((pred_cls != cls) & (true_cls == cls)).sum().float()
        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / (tp + fn).clamp(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        metrics[f"derived_{name}_precision"] = precision.item()
        metrics[f"derived_{name}_recall"] = recall.item()
        metrics[f"derived_{name}_f1"] = f1.item()
        f1s.append(f1.item())

    if f1s:
        metrics["derived_macro_f1"] = sum(f1s) / len(f1s)

    return metrics


# ======================================================================
# DataLoader factory
# ======================================================================

def load_dataloaders(
    path: str | Path,
    norm_mode: str = "raw",
    max_chunk: int = MAX_CHUNK,
    batch_size: int = 64,
    num_workers: int = 0,
    loss_oversample: float = 2.0,
    coverage: str = "full",
    shuffle: bool = True,
) -> dict[str, dict]:
    """Load every ``.h5ad`` under ``path`` (a directory or a single file) as a ChrDataset + DataLoader.

    Each file is expected to be processed and self-describing: raw-count expression in ``X``, a
    ``chr_boundary`` column in ``.var``, and the copy-number ground truth and ``cn_type`` in ``.uns``
    (see docs/input_format.md). No dataset registry or split column is required; the caller decides
    which files are training vs. validation by passing different paths.

    Returns: ``{dataset_name: {"ds": ChrDataset, "loader": DataLoader}}``.
    """
    path = Path(path)
    files = sorted(path.glob("*.h5ad")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .h5ad files found under {path}")

    result: dict = {}
    for h5ad_path in files:
        try:
            dataset = ChrDataset(
                h5ad_path=h5ad_path,
                norm_mode=norm_mode,
                max_chunk=max_chunk,
                coverage=coverage,
            )
        except ValueError as e:
            logger.warning("Skipping %s: %s", h5ad_path.name, e)
            continue

        if shuffle and loss_oversample > 1.0 and coverage == "full":
            weights = dataset.get_sample_weights(loss_oversample)
            sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
            loader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=True, drop_last=False,
            )
        else:
            if shuffle and loss_oversample > 1.0 and coverage == "random":
                dataset.get_sample_weights(loss_oversample)  # sets _chr_sample_probs
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True, drop_last=False,
            )

        result[dataset.dataset_name] = {"ds": dataset, "loader": loader}

    return result
