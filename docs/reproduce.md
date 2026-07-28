# Reproducing the paper's per-cell numbers

The released code reproduces the paper's reference-free per-cell copy-number results on the two paired
single-cell-DNA cohorts bundled with the data release (A375 and HCT116). "REFCON" throughout the paper
is the 3-checkpoint ensemble (`ens3` = context lengths 512 + 1024 + 2048), i.e. the per-cell mean of
the three checkpoints' `predicted_cn_ratios`.

## Run
```bash
# A375 (DNTR-seq melanoma; 64 cells with per-cell GT)
python scripts/infer.py --data data/a375/a375_dntr_per_cell.h5ad --out out/a375 --device cuda:0
python scripts/score.py --pred out/a375/ens3_cn.npz --gt data/a375/a375_dntr_per_cell_cn.npy

# HCT116 (DNTR-seq colorectal; 1,467 evaluable cells)
python scripts/infer.py --data data/hct116/hct116_dntr_per_cell.h5ad --out out/hct116 --device cuda:0
python scripts/score.py --pred out/hct116/ens3_cn.npz --gt data/hct116/hct116_dntr_per_cell_cn.npy
```
`--device` accepts `cuda:N`, `mps` (Apple GPU), or `cpu` (auto-falls back to cpu on failure).
Inference and scoring are decoupled: `infer.py` writes `ens3_cn.npz`, and `score.py` evaluates it
against per-cell ground truth with the paper's metric code (`refcon/metrics.py`).

## Expected metrics (reference-free ensemble)
| cohort  | n cells | Pearson | Spearman | AUROC loss | AUROC gain |
|---------|--------:|--------:|---------:|-----------:|-----------:|
| A375    |      64 |  0.442  |  0.473   |   0.765    |   0.770    |
| HCT116  |   1,467 |  0.364  |  0.347   |   0.664    |   0.826    |

Scoring runs per cell over genes with a valid ground-truth call (`gt >= 0`; the per-cell GT uses `-1`
as a no-call sentinel, which is excluded). Predictions and metrics are identical across devices; on
Apple MPS a sub-0.002 float drift can appear relative to CUDA. Metric definitions and the ground-truth
convention are in [`input_format.md`](input_format.md).

## Runtime
`scripts/infer.py` writes `runtime.json` (device, per-checkpoint wall time, total, cells/s, peak
memory). The three ensemble members run one at a time, so GPU memory stays near a single member
(about 250 MB) and the ensemble fits on essentially any GPU; host RAM is about 1 to 1.5 GB.
Throughput is a few cells per second on a laptop-class GPU and scales linearly with cell count
(memory is cell-count-independent, batch-driven). Cost rises with more genes, since more genes mean
more sliding windows per cell.

## Optional: reference-included refinement
When a sample contains diploid cells, they can optionally sharpen the aneuploid profiles (an
enhancement, never a requirement). See `refcon/refine.py` (`--ref-col`, `--ref-label`, `--svd-k 3`).

## Downstream tumor/normal classification
```bash
python scripts/classify.py --data <sample>.h5ad --cn out/<sample>/ens3_cn.npz
```
Confident-diploid detection (markers on expression), then GMM clustering on the CN predictions, then
per-cluster Pearson labeling at T=0.75, with a pure-sample dispersion fallback at T_sigma=0.135.
