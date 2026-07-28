"""Per-cell metric computations.

For each cell, computes Pearson, Spearman, AUROC_loss, AUROC_gain against
absolute-CN ground truth.  Returns both:
  - raw per-cell value arrays (NaN where not evaluable), one per metric
  - summary statistics (mean, std, min, max, n_evaluable) over the per-cell
    distribution.

Conventions:
  * absolute-CN GT thresholds: loss = (gt < 1.5), gain = (gt > 2.5)
  * AUROC scores: -pred for loss, +pred for gain
  * Per-cell evaluability filters:
      Pearson/Spearman: >=50 valid (pred, gt) pairs and both stds > 1e-8
      AUROC: >=10 positive AND >=10 negative cells in the binary label

Inputs are (n_cells, n_genes) numpy arrays.  NaN entries are treated as
"missing" and excluded from each cell's pair set.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

CN_LOSS_T = 1.5
CN_GAIN_T = 2.5
MIN_VALID_PAIRS = 50
MIN_AUROC_CLASS = 10


@dataclass
class PerCellResult:
    P: np.ndarray   # Pearson, NaN where not evaluable
    S: np.ndarray   # Spearman
    AL: np.ndarray  # AUROC loss (gt < 1.5; score = -pred)
    AG: np.ndarray  # AUROC gain (gt > 2.5; score = +pred)
    PRL: np.ndarray  # AUPRC loss (rank-based, scale-invariant)
    PRG: np.ndarray  # AUPRC gain
    n_valid_pairs: np.ndarray  # int per cell


def per_cell_metrics(pred: np.ndarray, gt: np.ndarray,
                     valid_mask: np.ndarray | None = None,
                     fill_diploid: bool = True) -> PerCellResult:
    """Compute per-cell P, S, AL, AG.

    Parameters
    ----------
    pred : (n_cells, n_genes) - tool's prediction (any monotonic-increasing
           function of CN; NaN where missing).
    gt : (n_cells, n_genes) - absolute-CN ground truth (NaN/-1 where missing).
    valid_mask : optional (n_cells, n_genes) bool - additional gate; if None,
                 derived from `gt` being finite and >= 0.
    fill_diploid : if True (default), evaluate on the FULL set of valid-GT
                   genes per cell. NaN predictions are imputed with that
                   cell's per-tool "diploid" baseline (= median of the cell's
                   finite predictions). This is the genome-wide convention -
                   penalizes low-coverage tools by treating uncovered genes
                   as predicting "neutral CN".
                   If False, restrict to genes where BOTH pred AND gt are
                   finite (partial-coverage convention; favors low-coverage
                   tools that only predict on "easy" genes).

    Returns
    -------
    PerCellResult with per-cell numpy arrays.  NaN where the metric was not
    evaluable for that cell.
    """
    n_cells = pred.shape[0]
    if valid_mask is None:
        valid_mask = np.isfinite(gt) & (gt >= 0)

    P = np.full(n_cells, np.nan, dtype=np.float64)
    S = np.full(n_cells, np.nan, dtype=np.float64)
    AL = np.full(n_cells, np.nan, dtype=np.float64)
    AG = np.full(n_cells, np.nan, dtype=np.float64)
    PRL = np.full(n_cells, np.nan, dtype=np.float64)
    PRG = np.full(n_cells, np.nan, dtype=np.float64)
    n_valid = np.zeros(n_cells, dtype=np.int64)

    for c in range(n_cells):
        gt_finite = valid_mask[c] & np.isfinite(gt[c])
        pred_finite = np.isfinite(pred[c])

        if fill_diploid:
            m = gt_finite
            n = int(m.sum())
            n_valid[c] = n
            if n < MIN_VALID_PAIRS:
                continue
            # Per-cell diploid baseline = median of finite predictions
            # across the cell's GT-valid positions.
            both = gt_finite & pred_finite
            if both.sum() < MIN_VALID_PAIRS:
                continue
            diploid = float(np.median(pred[c][both]))
            p_c = pred[c].copy().astype(np.float64)
            p_c[~pred_finite] = diploid
            p = p_c[m]
            g = gt[c][m].astype(np.float64)
        else:
            m = gt_finite & pred_finite
            n = int(m.sum())
            n_valid[c] = n
            if n < MIN_VALID_PAIRS:
                continue
            p = pred[c][m].astype(np.float64)
            g = gt[c][m].astype(np.float64)

        # Pearson
        ps, gs = p.std(), g.std()
        if ps > 1e-8 and gs > 1e-8:
            try:
                P[c] = float(np.corrcoef(p, g)[0, 1])
            except Exception:
                pass

        # Spearman: tied-aware rank (method='average') then Pearson on ranks.
        # Argsort-of-argsort breaks ties by index order; for predictions with
        # heavy ties (e.g. inferCNVpy with ~100 unique values across 10k genes)
        # this artificially aligns gene-index ordering of pred with chr-position
        # ordering of GT, giving spuriously high Spearman.
        ra = rankdata(p, method="average")
        rb = rankdata(g, method="average")
        if ra.std() > 1e-8 and rb.std() > 1e-8:
            try:
                S[c] = float(np.corrcoef(ra, rb)[0, 1])
            except Exception:
                pass

        # AUROC loss / gain
        gt_loss = (g < CN_LOSS_T).astype(int)
        gt_gain = (g > CN_GAIN_T).astype(int)
        n_loss = int(gt_loss.sum())
        n_gain = int(gt_gain.sum())
        if n_loss >= MIN_AUROC_CLASS and (n - n_loss) >= MIN_AUROC_CLASS:
            try:
                AL[c] = float(roc_auc_score(gt_loss, -p))
            except Exception:
                pass
            try:
                PRL[c] = float(average_precision_score(gt_loss, -p))
            except Exception:
                pass
        if n_gain >= MIN_AUROC_CLASS and (n - n_gain) >= MIN_AUROC_CLASS:
            try:
                AG[c] = float(roc_auc_score(gt_gain, p))
            except Exception:
                pass
            try:
                PRG[c] = float(average_precision_score(gt_gain, p))
            except Exception:
                pass

    return PerCellResult(P=P, S=S, AL=AL, AG=AG, PRL=PRL, PRG=PRG,
                         n_valid_pairs=n_valid)


def per_line_metrics(pred: np.ndarray, gt_per_cell: np.ndarray,
                      cell_lines: np.ndarray,
                      valid_mask: np.ndarray | None = None,
                      fill_diploid: bool = True) -> dict:
    """Per-cell-line CN profiling evaluation.

    For each cell line:
      pseudobulk_pred = nan-aware mean of pred over cells of that line
      line_gt = the line's per-cell GT (all cells of one line share the same
                bulk-DepMap GT row, so any cell's GT row is the line's GT)
      Compute Pearson, Spearman, AUROC_loss, AUROC_gain.

    Parameters
    ----------
    pred : (n_cells, n_genes) float - tool's predictions (NaN where missing)
    gt_per_cell : (n_cells, n_genes) - absolute-CN GT for each cell
    cell_lines : (n_cells,) str - line label per cell
    valid_mask : optional (n_cells, n_genes) bool - additional gate
    fill_diploid : if True, fill pred NaN with per-cell median before averaging
                   (matches per_cell_metrics convention).

    Returns
    -------
    dict[line] = {"P", "S", "AL", "AG", "n_cells_in_line", "n_valid_pairs"}
    """
    if valid_mask is None:
        valid_mask = np.isfinite(gt_per_cell) & (gt_per_cell >= 0)

    pred_use = pred.astype(np.float32, copy=True)
    if fill_diploid:
        for c in range(pred_use.shape[0]):
            fin = np.isfinite(pred_use[c])
            if fin.any():
                med = float(np.median(pred_use[c, fin]))
                pred_use[c, ~fin] = med

    out = {}
    unique_lines = sorted(set(str(x) for x in cell_lines))
    for line in unique_lines:
        m_cells = np.asarray([str(x) == line for x in cell_lines])
        if m_cells.sum() == 0:
            continue
        # Pseudobulk: nan-aware mean across cells of this line
        sub_pred = pred_use[m_cells]
        with np.errstate(all="ignore"):
            pb_pred = np.nanmean(sub_pred, axis=0)
        # Line GT: take any cell's GT row (they're all equal for bulk-DepMap projection)
        # Use the first cell's GT.
        gt_idxs = np.where(m_cells)[0]
        line_gt = gt_per_cell[gt_idxs[0]]
        line_valid = valid_mask[gt_idxs[0]]

        m = line_valid & np.isfinite(pb_pred) & np.isfinite(line_gt)
        n = int(m.sum())
        result = {"n_cells_in_line": int(m_cells.sum()), "n_valid_pairs": n,
                  "P": np.nan, "S": np.nan, "AL": np.nan, "AG": np.nan,
                  "PRL": np.nan, "PRG": np.nan}

        if n >= MIN_VALID_PAIRS:
            p = pb_pred[m].astype(np.float64)
            g = line_gt[m].astype(np.float64)
            if p.std() > 1e-8 and g.std() > 1e-8:
                result["P"] = float(np.corrcoef(p, g)[0, 1])
                ra = rankdata(p, method="average")
                rb = rankdata(g, method="average")
                if ra.std() > 1e-8 and rb.std() > 1e-8:
                    result["S"] = float(np.corrcoef(ra, rb)[0, 1])
            gt_loss = (g < CN_LOSS_T).astype(int)
            gt_gain = (g > CN_GAIN_T).astype(int)
            n_loss = int(gt_loss.sum()); n_gain = int(gt_gain.sum())
            if n_loss >= MIN_AUROC_CLASS and (n - n_loss) >= MIN_AUROC_CLASS:
                try: result["AL"] = float(roc_auc_score(gt_loss, -p))
                except Exception: pass
                try: result["PRL"] = float(average_precision_score(gt_loss, -p))
                except Exception: pass
            if n_gain >= MIN_AUROC_CLASS and (n - n_gain) >= MIN_AUROC_CLASS:
                try: result["AG"] = float(roc_auc_score(gt_gain, p))
                except Exception: pass
                try: result["PRG"] = float(average_precision_score(gt_gain, p))
                except Exception: pass
        out[line] = result
    return out


def summarize(values: np.ndarray) -> dict:
    """Mean / std / min / max / n_evaluable, ignoring NaN."""
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"n": n,
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max())}
