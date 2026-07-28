#!/usr/bin/env python3
"""Cell-wide CN ratio inference for BinnedDevModel.
Chain: cellwide_direct (joint log-space LSQ over chr-internal + bridge windows).

Per cell, produces a genome-wide CN-ratio profile (normalized so the
cell's mean over valid bins/genes equals 1). The pipeline is:

  1. Bidirectional window forward at two strides:
       - chr-internal windows  (stride = max_chunk // 2)
       - chr-boundary bridges  (stride = max_chunk // 4)
     "Bidirectional" = average the model output on the window with its
     gene-order-reversed counterpart. This cancels the left-low / right-
     high positional asymmetry the model learned from its within-chr
     training distribution.
  2. cellwide_direct chain (`chain_cellwide_direct`): one joint log-space
     LSQ over ALL windows simultaneously (chr-internal + bridges). For
     every pair of windows i, j sharing one or more global bins g, append
        x_j - x_i = log( r_i[g] / r_j[g] )
     plus an anchor x_0 = 0 (high weight). Solve for per-window log-scales
     x; rescale window predictions by exp(-x); take the mean over windows
     covering each global bin; renormalize each cell to mean 1.

This script is self-contained - no imports from any other file in this
folder except the model definitions.

Usage
-----

    python -m refcon.eval_cellwide \\
        --checkpoint /path/to/baseline_bs8_mc1024.pt \\
        --data        /path/to/data.h5ad \\
        --output      /path/to/predictions.pt \\
        --device      cuda:0
        # optional: --eval  to compute metrics against `var["cn_ground_truth"]`
        #          or per-cell CN in `uns["cn_matrix"]` when present

Output .pt contains:
    cell_ids            : (n_cells,) str
    gene_symbols        : (n_genes,) str
    bin_coords          : list[(gene_start, gene_end)] per bin
    predicted_cn_ratios : (n_cells, n_genes) float32 - per-cell, cell-mean-1
    bin_profile         : (n_cells, n_bins)  float32 - compact bin version
    chr_ranges          : list[(gene_start, gene_end)] per chromosome
    chr_bin_map         : dict chr_idx -> list[bin_idx]
    config              : dict (bin_size, max_chunk, d_model, ...)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from .model_binned_dev import BinnedDevModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_cellwide")


# ===================================================================== #
# 1. Windows
# ===================================================================== #

def get_chr_ranges(adata):
    bounds = adata.var["chr_boundary"].values.astype(bool)
    starts = np.nonzero(bounds)[0]
    if len(starts) == 0 or starts[0] != 0:
        starts = np.concatenate([[0], starts])
    ends = list(starts[1:]) + [adata.n_vars]
    return list(zip(starts.tolist(), [int(e) for e in ends]))


def build_chr_windows(chr_ranges, max_chunk: int, bin_size: int, stride_div: int = 2):
    """stride = max_chunk // stride_div, strictly within each chromosome.
    Default stride_div=2 (release behavior - half-window stride).
    Returns list of (chr_idx, ws, we, n_usable_bins).
    """
    stride = max(max_chunk // stride_div, bin_size)
    out = []
    for ci, (cs, ce) in enumerate(chr_ranges):
        for ws in range(cs, ce, stride):
            we = min(ws + max_chunk, ce)
            n_usable = (we - ws) // bin_size
            if n_usable < 2:
                continue
            out.append((ci, ws, we, n_usable))
            if we == ce:
                break
    return out


def build_bridge_windows(chr_ranges, max_chunk: int, bin_size: int, stride_div: int = 4):
    """Windows of length max_chunk straddling each chr/chr boundary.
    stride across boundary = max_chunk // stride_div (default 4 - release behavior).
    Bins never cross the boundary (ws chosen so (boundary - ws) % bin_size == 0).
    Drop any window that does not carry at least 2 bins on each side.

    Returns list of (left_ci, right_ci, ws, we, n_bins_left, n_bins_right).
    """
    stride = max(max_chunk // stride_div, bin_size)
    out = []
    for i in range(len(chr_ranges) - 1):
        cs_i, ce_i = chr_ranges[i]
        cs_j, ce_j = chr_ranges[i + 1]
        boundary = ce_i
        align = boundary % bin_size
        min_ws = max(cs_i, boundary - max_chunk + bin_size)
        max_ws = min(ce_j - bin_size, boundary - bin_size)
        if max_ws < min_ws:
            continue
        first = min_ws + ((align - min_ws) % bin_size)
        for ws in range(first, max_ws + 1, stride):
            we = ws + max_chunk
            if we > ce_j:
                we = cs_j + ((ce_j - cs_j) // bin_size) * bin_size
            n_bins_left = (boundary - ws) // bin_size
            n_bins_right = (we - boundary) // bin_size
            if n_bins_left < 2 or n_bins_right < 2:
                continue
            out.append((i, i + 1, int(ws), int(we),
                        int(n_bins_left), int(n_bins_right)))
    return out


# ===================================================================== #
# 2. Bidirectional forward
# ===================================================================== #

def _reverse_prefix(chunk: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return chunk with chunk[i, :L_i] reversed per row; padding preserved."""
    max_chunk = chunk.shape[1]
    idx = torch.arange(max_chunk, device=chunk.device).unsqueeze(0).expand(chunk.shape[0], -1)
    L = lengths.unsqueeze(1)
    rev_idx = torch.where(idx < L, L - 1 - idx, idx)
    return torch.gather(chunk, 1, rev_idx)


@torch.no_grad()
def forward_window_bidir(model, chunk: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Bidirectional per-window forward: (fwd + reversed-un-reversed) / 2.

    Uses the fact that Conv1d(k=bin_size, s=bin_size) maps reversed-input
    bin k to original bin (nbv - 1 - k), where nbv = ceil(L / bin_size).
    """
    bp_fwd, bin_lengths = model(chunk, lengths)
    chunk_rev = _reverse_prefix(chunk, lengths)
    bp_rev, _ = model(chunk_rev, lengths)

    B, n_bins = bp_fwd.shape
    nbv = bin_lengths.to(bp_rev.device).long()
    k_idx = torch.arange(n_bins, device=bp_rev.device).unsqueeze(0)
    src = nbv.unsqueeze(1) - 1 - k_idx
    in_range = (src >= 0) & (src < nbv.unsqueeze(1))
    src_safe = torch.clamp(src, min=0, max=n_bins - 1)
    bp_rev_un = torch.gather(bp_rev, 1, src_safe)
    bp_avg = (bp_fwd + bp_rev_un) / 2.0
    return torch.where(in_range, bp_avg, bp_fwd)


@torch.no_grad()
def predict_windows_bidir(model, X_log: np.ndarray, windows,
                          pack_fn, device, cell_batch: int = 8) -> np.ndarray:
    """Run bidirectional forward on each window across all cells.
    `pack_fn(entry) -> (ws, usable_len)` tells us where to slice X_log.
    Returns (n_cells, n_windows, n_model_bins) float32 with NaN beyond usable bins.
    """
    n_cells = X_log.shape[0]
    max_chunk = model.max_chunk
    n_model_bins = model.n_bins
    preds = np.full((n_cells, len(windows), n_model_bins), np.nan, dtype=np.float32)

    for wi, entry in enumerate(windows):
        ws, usable_len = pack_fn(entry)
        chunk = np.zeros((n_cells, max_chunk), dtype=np.float32)
        chunk[:, :usable_len] = X_log[:, ws:ws + usable_len]
        lengths = np.full(n_cells, usable_len, dtype=np.int64)
        n_valid_bins = usable_len // model.bin_size
        for c0 in range(0, n_cells, cell_batch):
            c1 = min(c0 + cell_batch, n_cells)
            expr_t = torch.from_numpy(chunk[c0:c1]).to(device)
            len_t = torch.from_numpy(lengths[c0:c1]).to(device)
            bp = forward_window_bidir(model, expr_t, len_t)
            preds[c0:c1, wi, :n_valid_bins] = bp[:, :n_valid_bins].cpu().numpy()
    return preds


# ===================================================================== #
# 3. Per-chromosome LSQ chain
# ===================================================================== #

def chain_baselines_lstsq(window_preds, window_starts, window_ends):
    """Joint log-space LSQ over overlapping bin pairs within one chromosome.

    For every overlap bin g between windows i and j:
        x_j - x_i = log(r_i[g] / r_j[g])
    where x_i = log(b_rel[i]). Solved via numpy.linalg.lstsq with anchor
    x_0 = 0 (high weight). Returns `b_rel` normalized so mean(b_rel) = 1.
    """
    n_win = len(window_starts)
    if n_win < 2:
        return np.ones(n_win, dtype=np.float64)

    rows = []
    for i in range(n_win):
        s_a, e_a = window_starts[i], window_ends[i]
        for j in range(i + 1, n_win):
            s_b, e_b = window_starts[j], window_ends[j]
            if s_b >= e_a:
                break
            overlap_start = max(s_a, s_b); overlap_end = min(e_a, e_b)
            if overlap_end <= overlap_start:
                continue
            r_a = window_preds[i, overlap_start - s_a : overlap_end - s_a]
            r_b = window_preds[j, overlap_start - s_b : overlap_end - s_b]
            m = np.isfinite(r_a) & np.isfinite(r_b) & (r_a > 1e-3) & (r_b > 1e-3)
            if m.sum() < 1:
                continue
            ra = np.clip(r_a[m].astype(np.float64), 0.25, 4.0)
            rb = np.clip(r_b[m].astype(np.float64), 0.25, 4.0)
            ld = np.log(ra) - np.log(rb)
            for v in ld:
                rows.append((i, j, float(v)))

    if not rows:
        return np.ones(n_win, dtype=np.float64)

    n_eq = len(rows) + 1
    A = np.zeros((n_eq, n_win), dtype=np.float64)
    bvec = np.zeros(n_eq, dtype=np.float64)
    for k, (i, j, ld) in enumerate(rows):
        A[k, i] = -1.0; A[k, j] = 1.0; bvec[k] = ld
    A[-1, 0] = np.sqrt(1e3); bvec[-1] = 0.0

    try:
        x, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return np.ones(n_win, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return np.ones(n_win, dtype=np.float64)
    x = np.clip(x, np.log(0.1), np.log(10.0))
    b_rel = np.exp(x - x.mean())
    mean_br = float(np.mean(b_rel))
    return b_rel / mean_br if mean_br > 1e-6 else b_rel


def per_chr_chain(all_win_preds, windows, chr_ranges, bin_size):
    """Aggregate chr-internal window preds into a bin-level chained profile.

    Returns:
        bin_profile : (n_cells, n_total_bins)
        bin_coords  : list of (gs, ge) per global bin
        chr_bin_map : dict chr_idx -> list[global bin indices]
    """
    n_cells = all_win_preds.shape[0]
    bin_set = set()
    for ci, ws, we, n_ub in windows:
        for b in range(n_ub):
            bin_set.add((ws + b * bin_size, ws + (b + 1) * bin_size))
    bin_coords = sorted(bin_set)
    bin_to_idx = {b: i for i, b in enumerate(bin_coords)}
    n_total = len(bin_coords)

    chr_to_wins = {}
    for w_idx, (ci, ws, we, n_ub) in enumerate(windows):
        chr_to_wins.setdefault(ci, []).append(w_idx)

    bin_profile = np.full((n_cells, n_total), np.nan, dtype=np.float32)
    for cell in range(n_cells):
        for ci, win_idxs in chr_to_wins.items():
            if not win_idxs:
                continue
            ws_bin, we_bin, preds_list = [], [], []
            for wi in win_idxs:
                _, ws, we, n_ub = windows[wi]
                first = bin_to_idx[(ws, ws + bin_size)]
                ws_bin.append(first); we_bin.append(first + n_ub)
                preds_list.append(all_win_preds[cell, wi, :n_ub])
            ws_arr = np.array(ws_bin, dtype=np.int64)
            we_arr = np.array(we_bin, dtype=np.int64)
            max_nb = max(we_arr - ws_arr)
            preds_padded = np.full((len(win_idxs), max_nb), np.nan, dtype=np.float32)
            for k, p in enumerate(preds_list):
                preds_padded[k, :len(p)] = p
            b_rel = chain_baselines_lstsq(preds_padded, ws_arr, we_arr)
            for k in range(len(win_idxs)):
                _, ws, we, n_ub = windows[win_idxs[k]]
                for b_local in range(n_ub):
                    g_bin = ws_arr[k] + b_local
                    if np.isnan(bin_profile[cell, g_bin]):
                        r = float(preds_padded[k, b_local])
                        if np.isfinite(r) and r > 1e-3:
                            bin_profile[cell, g_bin] = b_rel[k] * r

    chr_bin_map = {}
    for bi, (bs, be) in enumerate(bin_coords):
        for ci_chr, (cs, ce) in enumerate(chr_ranges):
            if cs <= bs < ce:
                chr_bin_map.setdefault(ci_chr, []).append(bi)
                break
    return bin_profile, bin_coords, chr_bin_map


# ===================================================================== #
# 4. Cell-wide bridge LSQ
# ===================================================================== #

def cell_wide_scale_from_bridges(bridge_preds_cell: np.ndarray, bridges, n_chr: int):
    """Solve one log-scale s_c per chromosome from bridge predictions.
    Equations:  s_{ci_l} - s_{ci_r}  =  mean(log r_left) - mean(log r_right)
    Anchor s_0 = 0, center so mean(s) = 0. Returns (n_chr,) scales (exp s).
    """
    rows = []
    for bi, (ci_l, ci_r, ws, we, nbl, nbr) in enumerate(bridges):
        r = bridge_preds_cell[bi]
        rl, rr = r[:nbl], r[nbl:nbl + nbr]
        m_l = np.isfinite(rl) & (rl > 1e-3)
        m_r = np.isfinite(rr) & (rr > 1e-3)
        if m_l.sum() < 2 or m_r.sum() < 2:
            continue
        ra = np.clip(rl[m_l].astype(np.float64), 0.25, 4.0)
        rb = np.clip(rr[m_r].astype(np.float64), 0.25, 4.0)
        d = float(np.mean(np.log(ra)) - np.mean(np.log(rb)))
        w = float(min(m_l.sum(), m_r.sum()))
        rows.append((ci_l, ci_r, d, w))

    if not rows:
        return np.ones(n_chr, dtype=np.float64)

    n_eq = len(rows) + 1
    A = np.zeros((n_eq, n_chr), dtype=np.float64)
    bvec = np.zeros(n_eq, dtype=np.float64)
    for k, (ci_l, ci_r, d, w) in enumerate(rows):
        sqw = np.sqrt(w)
        A[k, ci_l] = sqw; A[k, ci_r] = -sqw; bvec[k] = d * sqw
    A[-1, 0] = np.sqrt(1e3); bvec[-1] = 0.0
    try:
        x, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return np.ones(n_chr, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return np.ones(n_chr, dtype=np.float64)
    x = np.clip(x, np.log(0.25), np.log(4.0))
    x = x - x.mean()
    return np.exp(x)


def apply_cell_wide_scales(bin_profile_chr, chr_bin_map, scales):
    """Multiply each chr's bins by exp(s_c), renormalize each cell to mean 1."""
    n_cells, n_bins = bin_profile_chr.shape
    out = np.full_like(bin_profile_chr, np.nan, dtype=np.float32)
    for c in range(n_cells):
        row = bin_profile_chr[c].copy()
        for ci, bi_list in chr_bin_map.items():
            ba = np.array(bi_list, dtype=np.int64)
            row[ba] = row[ba] * scales[c, ci]
        v = np.isfinite(row) & (row > 1e-6)
        if v.sum() < 2:
            continue
        m = float(np.mean(row[v]))
        if m < 1e-6:
            continue
        out[c] = row / m
    return out


# ===================================================================== #
# 5. Driver
# ===================================================================== #

def broadcast_bins_to_genes(bin_profile, bin_coords, n_genes):
    n_cells = bin_profile.shape[0]
    out = np.full((n_cells, n_genes), np.nan, dtype=np.float32)
    for bi, (gs, ge) in enumerate(bin_coords):
        if gs < n_genes:
            out[:, gs:min(ge, n_genes)] = bin_profile[:, bi:bi + 1]
    return out


def chain_cellwide_direct(chr_preds, bridge_preds, chr_windows, chr_ranges,
                           bridges, bin_size):
    """Joint log-space LSQ across ALL windows (chr-internal + bridge), per cell.

    For every pair of windows i, j sharing one or more global bins g, the LSQ
    enforces  x_j - x_i ~ log( r_i[g] / r_j[g] ), plus an anchor x_0 = 0
    (high-weight). Per-window log-scales x are recovered, predictions are
    rescaled by exp(-x), and overlapping bins are averaged for the final
    profile (cell mean = 1).

    Implemented via normal equations (A^T A x = A^T b), built incrementally
    across overlap pairs without ever materialising A. This is exact (same
    solution as np.linalg.lstsq on the full system) and ~50-100x faster than
    the naive row-by-row build.

    Returns
    -------
    bin_profile : (n_cells, n_global_bins) float32 ratios (cell-mean ~ 1)
    bin_coords  : list[(gs, ge)]
    chr_bin_map : dict chr_idx -> list[global bin indices]
    """
    n_cells = chr_preds.shape[0]

    # ---- Build global bin layout (chr-aligned, contiguous, bin_size) ----
    all_bin_starts, chr_bin_map = [], {}
    for ci, (cs, ce) in enumerate(chr_ranges):
        chr_bin_start = len(all_bin_starts)
        g = cs
        while g + bin_size <= ce:
            all_bin_starts.append(g)
            g += bin_size
        chr_bin_map[ci] = list(range(chr_bin_start, len(all_bin_starts)))
    n_global_bins = len(all_bin_starts)
    all_bin_starts_arr = np.asarray(all_bin_starts, dtype=np.int64)
    bin_coords = [(s, s + bin_size) for s in all_bin_starts]

    # ---- Assemble all windows: (source, window_idx, global_bin_start, n_bins) ----
    all_windows = []
    for wi, (ci, ws, we, nbv) in enumerate(chr_windows):
        gb_start = int(np.searchsorted(all_bin_starts_arr, ws))
        all_windows.append(("chr", wi, gb_start, nbv))
    for wi, (ci_l, ci_r, ws, we, nbl, nbr) in enumerate(bridges):
        gb_start = int(np.searchsorted(all_bin_starts_arr, ws))
        all_windows.append(("bridge", wi, gb_start, nbl + nbr))

    n_win = len(all_windows)
    bin_profile = np.full((n_cells, n_global_bins), np.nan, dtype=np.float32)

    # Pre-compute pairwise overlap layouts once (cell-independent).
    # Each entry: (i, j, idx_i, idx_j) - shared bin local indices in each window.
    overlaps = []
    for i in range(n_win):
        _, _, g0_i, nbv_i = all_windows[i]
        for j in range(i + 1, n_win):
            _, _, g0_j, nbv_j = all_windows[j]
            s = max(g0_i, g0_j); e = min(g0_i + nbv_i, g0_j + nbv_j)
            if e <= s:
                continue
            idx_i = np.arange(s - g0_i, e - g0_i, dtype=np.int64)
            idx_j = np.arange(s - g0_j, e - g0_j, dtype=np.int64)
            overlaps.append((i, j, idx_i, idx_j))

    # Original ablation uses anchor_w = 1e4 in row form, which becomes anchor_w**2 = 1e8
    # in the A^T A normal-equations form.
    anchor_w_sq = 1e8

    for c in range(n_cells):
        # Pull per-window predictions for this cell.
        win_preds = []
        for src, wi, gb0, nbv in all_windows:
            preds = chr_preds[c, wi, :nbv] if src == "chr" else bridge_preds[c, wi, :nbv]
            win_preds.append((gb0, nbv, preds))

        # Build normal equations  AtA x = Atb. Each equation row has -1 at col i,
        # +1 at col j, and rhs b = log(r_j / r_i). Anchor row puts anchor_w at col 0
        # with rhs 0, contributing anchor_w**2 to AtA[0,0].
        AtA = np.zeros((n_win, n_win), dtype=np.float64)
        Atb = np.zeros(n_win, dtype=np.float64)
        AtA[0, 0] += anchor_w_sq

        # Vectorise per overlap-pair.
        for i, j, idx_i, idx_j in overlaps:
            r_i = win_preds[i][2][idx_i]
            r_j = win_preds[j][2][idx_j]
            m = np.isfinite(r_i) & np.isfinite(r_j) & (r_i > 0) & (r_j > 0)
            if not m.any():
                continue
            ra = r_i[m].astype(np.float64)
            rb = r_j[m].astype(np.float64)
            # log_ratio_ij = log(r_i / r_j). Equation rhs is b_g = log(r_j/r_i) = -log_ratio_ij.
            log_ratio = np.log(ra) - np.log(rb)
            n_eq = log_ratio.size
            # AtA contribution: outer product of row (-1@i, +1@j) with itself, summed over n_eq equations.
            AtA[i, i] += n_eq
            AtA[j, j] += n_eq
            AtA[i, j] -= n_eq
            AtA[j, i] -= n_eq
            # Atb contribution: row^T * b. row[i]=-1 -> Atb[i] += -b = -log(r_j/r_i) = log(r_i/r_j) = log_ratio.
            #                              row[j]=+1 -> Atb[j] += +b = -log_ratio.
            s = float(log_ratio.sum())
            Atb[i] += s
            Atb[j] -= s

        # Solve  AtA x = Atb. Add tiny ridge for numerical stability if singular.
        try:
            x = np.linalg.solve(AtA + 1e-9 * np.eye(n_win), Atb)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(x)):
            continue
        scales = np.exp(x)  # (n_win,)

        # Apply scales: bin_profile[g] = mean over windows covering g of r_w[g] / exp(x_w).
        # Vectorise per-window: scatter into bin_sum / bin_cnt arrays.
        bin_sum = np.zeros(n_global_bins, dtype=np.float64)
        bin_cnt = np.zeros(n_global_bins, dtype=np.int32)
        for wi_idx, (gb0, nbv, preds) in enumerate(win_preds):
            scale = scales[wi_idx]
            if scale < 1e-6:
                continue
            preds_arr = np.asarray(preds, dtype=np.float64)
            valid = np.isfinite(preds_arr) & (preds_arr > 1e-3)
            if not valid.any():
                continue
            local_idx = np.nonzero(valid)[0]
            global_idx = gb0 + local_idx
            in_range = global_idx < n_global_bins
            local_idx = local_idx[in_range]
            global_idx = global_idx[in_range]
            np.add.at(bin_sum, global_idx, preds_arr[local_idx] / scale)
            np.add.at(bin_cnt, global_idx, 1)

        valid_bins = bin_cnt > 0
        if not valid_bins.any():
            continue
        prof = np.full(n_global_bins, np.nan, dtype=np.float32)
        prof[valid_bins] = (bin_sum[valid_bins] / bin_cnt[valid_bins]).astype(np.float32)
        m = float(np.nanmean(prof[valid_bins]))
        if m > 1e-6:
            prof[valid_bins] = prof[valid_bins] / m
        bin_profile[c] = prof

    return bin_profile, bin_coords, chr_bin_map


def run_cellwide_inference(model, X_log: np.ndarray, chr_ranges,
                           device, cell_batch: int,
                           chr_stride_div: int = 2,
                           bridge_stride_div: int = 4):
    """Run the canonical cellwide_direct chain end-to-end.

    1. Bidirectional chr-internal window forward
    2. Bidirectional bridge window forward
    3. Joint LSQ over all windows simultaneously (cellwide_direct)

    `chr_stride_div` and `bridge_stride_div` control window stride
    (stride = max_chunk // div). Defaults (2, 4) = release behavior.
    """
    max_chunk = model.max_chunk
    bin_size = model.bin_size
    n_chr = len(chr_ranges)
    n_cells = X_log.shape[0]

    # ---- 1. bidirectional chr-internal forward ----
    chr_windows = build_chr_windows(chr_ranges, max_chunk, bin_size,
                                     stride_div=chr_stride_div)
    logger.info("  %d chr-internal windows (stride=%d, div=%d)",
                len(chr_windows), max(max_chunk // chr_stride_div, bin_size), chr_stride_div)
    t0 = time.time()
    chr_preds = predict_windows_bidir(
        model, X_log, chr_windows,
        pack_fn=lambda e: (e[1], e[3] * bin_size),  # ws, usable_len
        device=device, cell_batch=cell_batch,
    )
    logger.info("  chr-internal bidir forward: %.1fs", time.time() - t0)

    # ---- 2. bidirectional bridge forward ----
    bridges = build_bridge_windows(chr_ranges, max_chunk, bin_size,
                                    stride_div=bridge_stride_div)
    logger.info("  %d bridge windows (stride=%d, div=%d)",
                len(bridges), max(max_chunk // bridge_stride_div, bin_size), bridge_stride_div)
    t0 = time.time()
    bridge_preds = predict_windows_bidir(
        model, X_log, bridges,
        pack_fn=lambda e: (e[2], (e[4] + e[5]) * bin_size),  # ws, usable_len
        device=device, cell_batch=cell_batch,
    )
    logger.info("  bridge bidir forward: %.1fs", time.time() - t0)

    # ---- 3. joint cellwide_direct LSQ ----
    t0 = time.time()
    bin_profile, bin_coords, chr_bin_map = chain_cellwide_direct(
        chr_preds, bridge_preds, chr_windows, chr_ranges, bridges, bin_size,
    )
    logger.info("  cellwide_direct joint LSQ: %.1fs  (%d global bins)",
                time.time() - t0, len(bin_coords))

    return {
        "bin_profile": bin_profile,
        "bin_coords": bin_coords,
        "chr_bin_map": chr_bin_map,
    }


# ---------------- optional eval against ground truth --------------- #

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman_fast(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def _per_cell_pearson(pred, gt, mask):
    n_cells = pred.shape[0]
    p = np.where(mask, pred, 0.0).astype(np.float64)
    g = np.where(mask, gt,   0.0).astype(np.float64)
    n = mask.sum(axis=1).astype(np.float64)
    mp = p.sum(axis=1) / np.maximum(n, 1)
    mg = g.sum(axis=1) / np.maximum(n, 1)
    dp = np.where(mask, p - mp[:, None], 0.0)
    dg = np.where(mask, g - mg[:, None], 0.0)
    num = (dp * dg).sum(axis=1)
    sp_ = np.sqrt((dp * dp).sum(axis=1))
    sg_ = np.sqrt((dg * dg).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.where((sp_ > 1e-12) & (sg_ > 1e-12), sp_ * sg_, np.nan)
    return np.where(n >= 50, r, np.nan)


def extract_gt_cn(adata: ad.AnnData, prefer_per_cell: bool = False) -> np.ndarray | None:
    """Return (n_cells, n_genes) GT CN or None if unavailable.  NaN->-1."""
    n_cells, n_genes = adata.n_obs, adata.n_vars
    cn_profiles = adata.uns.get("cn_profiles", {})
    cn_matrix = adata.uns.get("cn_matrix", None)
    cn_cell_ids = adata.uns.get("cn_cell_ids", None)
    var_bulk = (adata.var["cn_ground_truth"].values.astype(np.float32)
                if "cn_ground_truth" in adata.var.columns else None)

    has_pc = cn_matrix is not None and cn_cell_ids is not None
    id_to_row = {c: i for i, c in enumerate(cn_cell_ids)} if has_pc else {}

    if prefer_per_cell:
        if not has_pc:
            return None
        gt = np.full((n_cells, n_genes), -1.0, dtype=np.float32)
        for ci in range(n_cells):
            cid = adata.obs.index[ci]
            if cid in id_to_row:
                gt[ci] = cn_matrix[id_to_row[cid]].astype(np.float32)
        return np.where(np.isnan(gt), -1.0, gt)

    gt = np.zeros((n_cells, n_genes), dtype=np.float32)
    any_set = False
    for ci in range(n_cells):
        if "cn_profile" in adata.obs.columns and cn_profiles:
            pname = str(adata.obs["cn_profile"].values[ci])
            if pname in cn_profiles:
                gt[ci] = np.asarray(cn_profiles[pname], dtype=np.float32)
                any_set = True; continue
        if has_pc:
            cid = adata.obs.index[ci]
            if cid in id_to_row:
                gt[ci] = cn_matrix[id_to_row[cid]].astype(np.float32)
                any_set = True; continue
        if var_bulk is not None:
            gt[ci] = var_bulk; any_set = True
        else:
            gt[ci] = np.nan
    if not any_set:
        return None
    return np.where(np.isnan(gt), -1.0, gt)


def eval_metrics_gene(pred_gene, gt_cn):
    """Cell-normalized ratio GT + pearson / spearman / AUROC / per-cell pearson."""
    from sklearn.metrics import roc_auc_score
    valid = gt_cn > -0.5
    nv = valid.sum(axis=1)
    safe = np.where(valid, gt_cn, 0.0)
    cell_mean = safe.sum(axis=1) / np.maximum(nv, 1)
    ok_cell = (nv >= 50) & (cell_mean > 0.1)
    gt_ratio = np.where(valid,
                        gt_cn / np.where(cell_mean[:, None] > 0, cell_mean[:, None], 1.0),
                        np.nan).astype(np.float32)
    gt_ratio[~ok_cell] = np.nan

    bv = valid & np.isfinite(pred_gene) & np.isfinite(gt_ratio) & np.isfinite(gt_cn)
    p = pred_gene[bv].astype(np.float64)
    g = gt_ratio[bv].astype(np.float64)
    ga = gt_cn[bv].astype(np.float64)
    out = {"n_valid_pairs": int(bv.sum())}
    if p.size < 100:
        return out
    out["pooled_pearson"]  = _pearson(p, g)
    out["pooled_spearman"] = _spearman_fast(p, g)
    gt_loss = (ga < 1.5).astype(int); gt_gain = (ga > 2.5).astype(int)
    out["auroc_loss"] = (float(roc_auc_score(gt_loss, -p))
                         if gt_loss.sum() > 10 and (1 - gt_loss).sum() > 10
                         else float("nan"))
    out["auroc_gain"] = (float(roc_auc_score(gt_gain, p))
                         if gt_gain.sum() > 10 and (1 - gt_gain).sum() > 10
                         else float("nan"))
    rs = _per_cell_pearson(pred_gene, gt_ratio, bv)
    out["per_cell_pearson_mean"]   = float(np.nanmean(rs))
    out["per_cell_pearson_median"] = float(np.nanmedian(rs))
    out["per_cell_pearson_std"]    = float(np.nanstd(rs))
    out["n_cells_with_corr"]       = int(np.isfinite(rs).sum())
    return out


# ===================================================================== #
# main
# ===================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data",       required=True, help="processed raw-UMI .h5ad")
    ap.add_argument("--output",     required=True)
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--cell-batch", type=int, default=8)
    ap.add_argument("--eval",       action="store_true",
                    help="Compute metrics against GT if present in the h5ad")
    ap.add_argument("--per-cell-gt", action="store_true",
                    help="Prefer uns['cn_matrix'] over bulk cn_profile for GT")
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BinnedDevModel(
        bin_size=cfg["bin_size"], max_chunk=cfg["max_chunk"],
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4), dim_ff=cfg.get("dim_ff", 512),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Checkpoint: bin_size=%d max_chunk=%d params=%d",
                cfg["bin_size"], cfg["max_chunk"], model.count_parameters())

    adata = ad.read_h5ad(args.data)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    X = adata.X
    X_dense = X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    X_log = np.log1p(X_dense)
    chr_ranges = get_chr_ranges(adata)
    cell_ids = adata.obs.index.values.astype(str)
    gene_symbols = adata.var.index.values.astype(str)
    logger.info("Input: n_cells=%d  n_genes=%d  n_chr=%d",
                n_cells, n_genes, len(chr_ranges))

    result = run_cellwide_inference(
        model, X_log, chr_ranges, device, args.cell_batch,
    )

    pred_gene = broadcast_bins_to_genes(result["bin_profile"], result["bin_coords"], n_genes)

    out = {
        "cell_ids": cell_ids,
        "gene_symbols": gene_symbols,
        "bin_coords": result["bin_coords"],
        "chr_bin_map": {int(k): [int(b) for b in v] for k, v in result["chr_bin_map"].items()},
        "chr_ranges": chr_ranges,
        "bin_profile": result["bin_profile"],
        "predicted_cn_ratios": pred_gene,
        "config": cfg,
    }

    if args.eval:
        gt_cn = extract_gt_cn(adata, prefer_per_cell=args.per_cell_gt)
        if gt_cn is None:
            logger.warning("--eval requested but no GT found in h5ad; skipping metrics")
        else:
            m = eval_metrics_gene(pred_gene, gt_cn)
            out["metrics"] = m
            logger.info("Gene-level metrics (cell-normalized GT):")
            for k, v in m.items():
                if isinstance(v, float):
                    logger.info("  %-30s  %.4f", k, v)
                else:
                    logger.info("  %-30s  %s", k, v)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    logger.info("Saved predictions -> %s", out_path)


if __name__ == "__main__":
    main()
