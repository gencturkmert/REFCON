#!/usr/bin/env python3
"""Optional reference-cell refinement for REFCON copy-number ratio predictions.

REFCON is reference-free: every cell is predicted independently, and the model
is trained without any notion of reference cells. When a cohort does contain
diploid reference cells, this module applies a post-hoc refinement that uses
them to remove systematic bias from the predictions:

  1. Per-gene mean bias: reference cells should predict ratio = 1.0 everywhere
     (they are diploid), so their deviation from 1.0 is systematic model bias.
     Estimate that bias per gene from the reference cells and subtract it from
     all cells.

  2. SVD bias-mode removal: the residual reference variance left after step 1 is
     multi-gene structure that the per-gene mean did not capture. An SVD of
     (refs - 1) identifies the dominant bias directions in gene space; project
     those directions out of every cell's residual.

  3. Renormalize each cell to mean 1.

The refinement is entirely post-hoc and never touches the model. Default
configuration: svd_k=3, no winsorizing.

Usage (programmatic):

    from refcon.refine import calibrate_predictions
    # pred: (n_cells, n_genes) - REFCON ratio output
    # ref_mask: (n_cells,) bool - True for known diploid cells
    pred_refined = calibrate_predictions(pred, ref_mask, svd_k=3)

Usage (CLI):

    python -m refcon.refine \\
        --predictions path/to/eval_cellwide_output.pt \\
        --data path/to/cohort.h5ad \\
        --output path/to/refined.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def renormalize_to_cell_mean_1(pred: np.ndarray) -> np.ndarray:
    """Scale each row (cell) so that finite-mean equals 1.
    NaN entries are preserved; values <= 1e-6 are treated as missing."""
    out = pred.copy()
    for c in range(pred.shape[0]):
        row = out[c]
        v = np.isfinite(row) & (row > 1e-6)
        if v.sum() < 2:
            continue
        m = float(np.mean(row[v]))
        if m > 1e-6:
            out[c] = row / m
    return out


def calibrate_predictions(
    pred: np.ndarray,
    ref_mask: np.ndarray,
    svd_k: int = 3,
    winsor_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply post-hoc reference-cell refinement to REFCON predictions.

    Args:
        pred: (n_cells, n_genes) float array of per-cell CN ratio predictions
            from the reference-free pipeline. Each cell should have mean ~ 1.
        ref_mask: (n_cells,) bool - True for cells that are known diploid
            (reference cells). Must have at least 3 refs for the SVD step.
        svd_k: int - number of principal bias modes to remove from the ref
            residuals. k=3 is the default. Set svd_k=0 to skip SVD and apply
            only per-gene bias correction.
        winsor_range: optional (lo, hi) tuple for robust winsorizing of ref
            predictions before computing per-gene bias. Default: None (no
            clipping).

    Returns:
        Calibrated predictions with the same shape as `pred`. Each cell is
        renormalized to mean 1 over valid bins.
    """
    if ref_mask.sum() < 3:
        raise ValueError(f"ref_mask has only {ref_mask.sum()} True values; "
                         "need >=3 ref cells for SVD.")

    # Step 1: per-gene mean bias from ref cells
    refs = pred[ref_mask]
    if winsor_range is not None:
        lo, hi = winsor_range
        refs = np.clip(refs, lo, hi)
    bias = np.nanmean(refs, axis=0) - 1.0
    bias = np.where(np.isfinite(bias), bias, 0.0)

    # Step 2: subtract per-gene bias from all cells
    pred_debiased = pred - bias[None, :]
    pred_debiased = np.where(np.isfinite(pred), pred_debiased, np.nan)
    pred_debiased = renormalize_to_cell_mean_1(pred_debiased)

    if svd_k <= 0:
        return pred_debiased

    # Step 3: SVD of residual ref predictions -> top-k bias modes
    refs_res = pred_debiased[ref_mask] - 1.0
    refs_res = np.where(np.isfinite(refs_res), refs_res, 0.0)
    k = min(svd_k, refs_res.shape[0] - 1)
    _, _, Vt = np.linalg.svd(refs_res, full_matrices=False)
    V_k = Vt[:k]  # (k, n_genes)

    # Step 4: project bias modes out of all cells
    centered = pred_debiased - 1.0
    centered = np.where(np.isfinite(centered), centered, 0.0)
    centered_clean = centered - (centered @ V_k.T) @ V_k
    result = 1.0 + centered_clean
    result = np.where(np.isfinite(pred), result, np.nan)

    # Step 5: renormalize to cell mean 1
    return renormalize_to_cell_mean_1(result)


def extract_ref_mask_from_h5ad(h5ad_path: str | Path,
                                role_col: str = "cn_role",
                                ref_label: str = "reference") -> np.ndarray:
    """Extract the reference-cell mask from an h5ad's obs column.
    Defaults assume obs['cn_role'] == 'reference' marks the reference (diploid) cells."""
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    if role_col not in a.obs.columns:
        raise ValueError(f"Column '{role_col}' not found in h5ad obs. "
                         f"Available: {list(a.obs.columns)}")
    return (a.obs[role_col].values == ref_label)


def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc reference-cell refinement for REFCON outputs."
    )
    ap.add_argument("--predictions", required=True,
                    help="Path to .pt file from eval_cellwide.py")
    ap.add_argument("--data", required=True,
                    help="Path to input .h5ad (used to extract the reference-cell mask)")
    ap.add_argument("--ref-col", default="cn_role",
                    help="obs column identifying ref cells (default: cn_role)")
    ap.add_argument("--ref-label", default="reference",
                    help="Value in --ref-col marking ref cells (default: reference)")
    ap.add_argument("--output", required=True,
                    help="Path to save calibrated predictions (.pt)")
    ap.add_argument("--svd-k", type=int, default=3,
                    help="SVD principal components to remove (default: 3)")
    ap.add_argument("--winsor-lo", type=float, default=None,
                    help="Optional winsorize lower bound")
    ap.add_argument("--winsor-hi", type=float, default=None,
                    help="Optional winsorize upper bound")
    args = ap.parse_args()

    import torch
    print(f"Loading {args.predictions} ...", flush=True)
    d = torch.load(args.predictions, map_location="cpu", weights_only=False)
    pred = d["predicted_cn_ratios"]  # (n_cells, n_genes)
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    pred = np.asarray(pred, dtype=np.float32)
    print(f"  predictions shape: {pred.shape}", flush=True)

    print(f"Extracting ref mask from {args.data} ...", flush=True)
    ref_mask = extract_ref_mask_from_h5ad(args.data, args.ref_col, args.ref_label)
    print(f"  refs: {ref_mask.sum()} / {len(ref_mask)}", flush=True)

    winsor = None
    if args.winsor_lo is not None and args.winsor_hi is not None:
        winsor = (args.winsor_lo, args.winsor_hi)
    print(f"Calibrating: svd_k={args.svd_k}  winsor={winsor}", flush=True)
    calibrated = calibrate_predictions(pred, ref_mask, svd_k=args.svd_k,
                                        winsor_range=winsor)

    d["predicted_cn_ratios_calibrated"] = torch.from_numpy(calibrated)
    d["calibration_config"] = {"svd_k": args.svd_k, "winsor_range": winsor,
                                 "n_refs": int(ref_mask.sum())}
    torch.save(d, args.output)
    print(f"Saved calibrated predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
