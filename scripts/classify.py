#!/usr/bin/env python3
"""Downstream aneuploid/diploid classification on REFCON's reference-free CN predictions.

Runs the paper's downstream caller (Methods 'Identifying confident diploid cells',
'Clustering', 'Labeling'): marker-based confident-diploid detection on expression ->
GMM clustering of the remaining cells on the CN predictions -> per-cluster Pearson label
against the in-sample diploid reference (T=0.75), with a pure-sample dispersion fallback
(T_sigma=0.135) when a cohort has <2 confident-diploid cells.

Usage
  python scripts/classify.py --data sample.h5ad --cn out/sample/ens3_cn.npz
    (--cn is the ens3 output of scripts/infer.py; genes must align to the h5ad.)
Prints per-cell labels summary + info; writes labels.npy next to --cn.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                               # repo root -> `import refcon`
from refcon.downstream.classify import call_sample          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="raw-UMI .h5ad (expression, for marker detection)")
    ap.add_argument("--cn", required=True, help="ens3_cn.npz from scripts/infer.py")
    a = ap.parse_args()
    A = ad.read_h5ad(a.data)
    counts = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
    genes = np.asarray(A.var_names).astype(str)
    d = np.load(a.cn, allow_pickle=True)
    cn, cells, cgenes = d["cn"], d["cells"].astype(str), d["genes"].astype(str)
    # align CN genes to the h5ad gene order the marker detector uses
    gi = {g: i for i, g in enumerate(cgenes)}
    col = np.array([gi.get(g, -1) for g in genes])
    pred = np.full((cn.shape[0], len(genes)), np.nan)
    pred[:, col >= 0] = cn[:, col[col >= 0]]
    labels, info = call_sample(counts, genes, pred)
    n_aneu = int((labels == 1).sum()); n_dip = int((labels == 0).sum())
    print(f"status={info['status']} n_conf_diploid={info['n_conf']}  "
          f"aneuploid={n_aneu} diploid={n_dip}  info={info}")
    np.save(Path(a.cn).parent / "labels.npy", labels)


if __name__ == "__main__":
    main()
