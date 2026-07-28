"""Marker-anchored aneuploid/diploid caller - standalone method (one sample at a time).

    from pipeline import call_sample
    labels, info = call_sample(raw_counts, gene_names, ens3_pred)
        labels : (n_cells,) int  -> 0 diploid, 1 aneuploid, -1 if PURE (not labeled)
        info   : dict(n_conf, status, K, cluster_corrs)

Steps: markers -> PURE gate -> reference -> sub1 cluster the rest -> pearson label @0.75.
NO calibration (kills correlation). Impute before consensus/corr.
"""
import numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from .common import impute
from .markers import ssmww_confident_normal

CORR_THR = 0.75
MIN_CONF = 2      # require >=2 confident-diploid cells so the diploid reference + cluster r_c are
                  # computed from more than one cell (a 1-cell reference is too noisy/biased to
                  # correlate against). Only genuinely reference-free cohorts (0-1 conf-diploid
                  # cells) fall through to the sigma branch. Was 3; lowered 2026-07-12 so mixed
                  # samples with a thin diploid signal (e.g. tir_60, 2 conf cells) are clustered
                  # instead of blanket-called. Mirror in eval_per_patient.py + threshold_*.py.
PURE_THR = 0.135  # max-margin midpoint of the validation dispersion gap [0.127 diploid ceiling, 0.143 aneuploid floor=SNU449]; identical calls to 0.14 on all pure cohorts

def sub1(X, n_pcs=5):
    """1-level GMM/BIC clustering: PCA(5) -> StandardScale -> GMM(full) k=1..8 argmin BIC."""
    Xi = impute(X)
    npc = min(n_pcs, Xi.shape[1], max(2, Xi.shape[0]-1))
    pcs = PCA(max(npc, 2), svd_solver='full').fit_transform(Xi)
    Xc = StandardScaler().fit_transform(pcs[:, :npc])
    bics, gms = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k in range(1, min(8, Xc.shape[0]-1)+1):
            try:
                gm = GaussianMixture(k, covariance_type='full', n_init=5, random_state=0).fit(Xc)
                bics.append(gm.bic(Xc)); gms.append(gm)
            except Exception:
                bics.append(np.inf); gms.append(None)
    g = gms[int(np.argmin(bics))]
    return g.predict(Xc) if g is not None else np.zeros(Xi.shape[0], int)

def call_sample(raw_counts, gene_names, pred, corr_thr=CORR_THR):
    cn = ssmww_confident_normal(raw_counts, gene_names)['norm_mask'].astype(bool)
    nconf = int(cn.sum())
    p = impute(np.asarray(pred, float))
    if nconf < MIN_CONF:
        # PURE sample (no diploid anchor): soft SAMPLE-LEVEL call via absolute aneuploidy score
        # (mean per-cell gene-dispersion). LOW-CONFIDENCE, sample-level, platform-dependent band.
        score = float(np.mean(np.nanstd(p, axis=1)))
        call = 1 if score >= PURE_THR else 0
        return (np.full(len(p), call, int),
                dict(n_conf=nconf, status='PURE_soft', aneuploidy_score=round(score, 4),
                     soft_call=('tumor-ish' if call else 'normal-ish'), pure_thr=PURE_THR))
    reference = np.median(p[cn], axis=0)              # diploid pseudo-cell
    out = np.zeros(len(p), int)                       # conf normals -> 0 (diploid)
    rest = ~cn; ridx = np.where(rest)[0]
    lab = sub1(p[rest])
    corrs = {}
    for c in range(lab.max()+1):
        s = lab == c
        r = pearsonr(np.median(p[rest][s], axis=0), reference)[0]
        corrs[int(c)] = float(r)
        out[ridx[s]] = 0 if r >= corr_thr else 1      # >=thr diploid, else aneuploid
    return out, dict(n_conf=nconf, status='labeled', K=int(lab.max()+1), cluster_corrs=corrs)
