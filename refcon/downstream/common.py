"""Shared helper for the downstream callers."""
import numpy as np


def impute(X):
    """Fill NaN with the per-column (per-gene) mean; columns that are all-NaN become 1.0."""
    X = np.asarray(X, dtype=np.float64).copy()
    if np.isnan(X).any():
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isfinite(cm), cm, 1.0)
        nm = np.isnan(X)
        X[nm] = np.take(cm, np.where(nm)[1])
    return X
