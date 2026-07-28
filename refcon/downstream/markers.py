"""Faithful Python port of SCEVAN's confident-normal (marker) detection.

Ported from SCEVAN R package (De Falco et al. 2023, Nat Commun):
  preprocessingMtx -> getConfidentNormalCells -> ssMwwGst (yaGST::mwwGST) -> top30classification

Pipeline:
  1. filter cells with >200 expressed genes
  2. filter genes expressed in >perc_genes (10%) of cells  (5% if low quality)
  3. ssMWW per cell:
       - z-score each gene ACROSS cells (standardize=TRUE)  <-- the key step
       - rank genes within the cell
       - per signature: Mann-Whitney U of signature-gene ranks vs the rest
         nes = U/(n1*n2) (= AUC),  NES = log2(nes/(1-nes)),  two-sided p
       - FDR = Benjamini-Hochberg across signatures, per cell
  4. confident normal = cells with, for >=1 signature, NES>=nes_cutoff(1) AND FDR<pval_cutoff(1e-10)
     (SCEVAN caps at top 30; we expose `cap` - default None = keep all passing, which is
      what we want for per-cluster fractions.)

The z-score-across-cells makes this a RELATIVE detector: it finds cells that look
normal *relative to other cells in the sample*. Pure-tumor and pure-normal samples
therefore yield few/no confident normals (no contrast) - by design.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parent
GENESET = json.load(open(HERE / 'scevan_geneset.json'))


def preprocess(counts, genes, perc_genes=0.10, min_genes_per_cell=200):
    """counts: (cells, genes) raw counts. genes: gene-name array.
    Returns (filtered_counts (cells,genes), kept_genes, kept_cell_mask).
    """
    counts = np.asarray(counts, dtype=np.float64)
    genes = np.asarray(genes, dtype=object)
    # 1) cells with >200 expressed genes
    g_per_cell = (counts > 0).sum(axis=1)
    cell_mask = g_per_cell > min_genes_per_cell
    if cell_mask.sum() == 0:
        cell_mask = g_per_cell > 0  # fallback
    c = counts[cell_mask]
    # 2) genes expressed in > perc_genes of cells
    der = (c > 0).sum(axis=0) / c.shape[0]
    pg = perc_genes if (der > perc_genes).sum() > 7000 else (perc_genes - 0.05)
    gene_mask = der > pg
    return c[:, gene_mask], genes[gene_mask], cell_mask


def ssmww_confident_normal(counts, genes, geneset=GENESET, min_len=5,
                           nes_cutoff=1.0, fdr_cutoff=1e-10, cap=None,
                           do_preprocess=True):
    """Returns dict:
        norm_mask   : (n_input_cells,) bool - confident-normal cells (in ORIGINAL cell order)
        best_sig    : (n_input_cells,) signature name for normal cells, '' otherwise
        n_confident : int
    """
    counts = np.asarray(counts, dtype=np.float64)
    genes = np.asarray(genes, dtype=object)
    n_in = counts.shape[0]

    if do_preprocess:
        fc, fg, cell_mask = preprocess(counts, genes)
    else:
        fc, fg, cell_mask = counts, genes, np.ones(n_in, bool)

    n_cells, n_genes = fc.shape
    gene_pos = {g: i for i, g in enumerate(fg)}
    sig_cols = {}
    for name, gl in geneset.items():
        idx = [gene_pos[g] for g in gl if g in gene_pos]
        if len(idx) >= min_len:
            sig_cols[name] = np.array(idx)
    sig_names = list(sig_cols.keys())

    # z-score each gene across cells (standardize=TRUE)
    mean = fc.mean(axis=0)
    sd = fc.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (fc - mean) / sd                      # (cells, genes)

    NES = np.full((len(sig_names), n_cells), np.nan)
    P = np.full((len(sig_names), n_cells), np.nan)

    for ci in range(n_cells):
        ranks = rankdata(Z[ci])               # ascending ranks over genes
        for si, name in enumerate(sig_names):
            cols = sig_cols[name]
            n1 = len(cols); n2 = n_genes - n1
            R1 = ranks[cols].sum()
            U = R1 - n1 * (n1 + 1) / 2.0       # MWW U for signature genes
            nes = U / (n1 * n2)                # AUC = P(sig rank > non-sig rank)
            nes = min(max(nes, 1e-9), 1 - 1e-9)
            NES[si, ci] = np.log2(nes / (1 - nes))
            mu = n1 * n2 / 2.0
            sig = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
            zstat = (U - mu) / sig
            P[si, ci] = 2.0 * norm.sf(abs(zstat))

    # BH FDR per cell (across signatures)
    FDR = np.full_like(P, np.nan)
    m = P.shape[0]
    for ci in range(n_cells):
        p = P[:, ci]
        order = np.argsort(p)
        ranked = p[order]
        adj = ranked * m / (np.arange(1, m + 1))
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty_like(adj)
        out[order] = np.clip(adj, 0, 1)
        FDR[:, ci] = out

    # top30classification logic
    FDRf = FDR.copy(); FDRf[FDRf >= fdr_cutoff] = 1.0
    NESf = NES.copy(); NESf[NESf < nes_cutoff] = 0.0
    comb = NESf * -np.log10(FDRf + 1e-300)
    score = comb.max(axis=0)                  # per filtered-cell
    cand = np.where(score > 0)[0]
    if cap is not None and len(cand) > cap:
        cand = cand[np.argsort(-score[cand])[:cap]]
    best = np.array([sig_names[comb[:, ci].argmax()] if score[ci] > 0 else '' for ci in range(n_cells)], dtype=object)

    # map back to original cell order
    norm_mask_filt = np.zeros(n_cells, bool); norm_mask_filt[cand] = True
    norm_mask = np.zeros(n_in, bool)
    best_full = np.array([''] * n_in, dtype=object)
    filt_idx = np.where(cell_mask)[0]
    norm_mask[filt_idx[cand]] = True
    best_full[filt_idx] = best
    return {'norm_mask': norm_mask, 'best_sig': best_full, 'n_confident': int(norm_mask.sum())}


__all__ = ['ssmww_confident_normal', 'preprocess', 'GENESET']
