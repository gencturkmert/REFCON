# REFCON data

The data objects are hosted on **Zenodo** (too large for git) and fetched by
`python scripts/fetch_data.py`. This directory holds the held-out evaluation cohorts used in the
paper's reference-free copy-number benchmark (Section 2.2). Every h5ad has **raw UMI counts** in `X`
and genomic annotations in `var` (`chromosome`, `start`, `end`, `chr_boundary`); genes are ordered by
genomic position. REFCON is reference-free, so no normal/reference cells are required at inference.

The cohorts fall into two ground-truth tiers.

## Tier 1  -  paired single-cell DNA (per-cell scWGS ground truth)
Each cell is scored against its **own** single-cell whole-genome DNA profile. This is the honest
per-cell verification of reference-free copy number.

### `a375/`  -  A375 melanoma (DNTR-seq; Zachariadis et al. 2020, GSE144296)
- `a375_dntr_per_cell.h5ad`  -  96 cells x 12,506 genes, raw UMI.
- `a375_dntr_per_cell_cn.npy`  -  `(96, 12506)` AneuFinder per-cell absolute CN, row-aligned to the
  h5ad. 64 cells evaluable; 32 (the HCA00101 plate) are all-NaN and excluded from scoring.

### `hct116/`  -  HCT116 colorectal (DNTR-seq; GSE144296)
- `hct116_dntr_per_cell.h5ad`  -  1,468 cells x 12,506 genes, raw UMI.
- `hct116_dntr_per_cell_cn.npy`  -  `(1468, 12506)` per-cell absolute CN, row-aligned (NaN where a
  cell has no GT); 1,467 evaluable.

### `scone/`  -  scONE-seq astrocytoma (Yu et al. 2023, GSE185269)
- `scone_astrocytoma.h5ad`  -  840 cells x 14,509 genes, raw UMI. `obs['cn_role']` splits the cells
  into **390 tumor** and **450 reference** (non-malignant); `obs['Clone_type']` gives the scWGS clone.
  Per-cell scWGS CN is also embedded in `uns['cn_matrix']` (rows = `uns['cn_cell_ids']`). The 450
  reference cells support the reference-included analysis (Section 2.5).
- `scone_astrocytoma_cn.npy`  -  `(840, 14509)` per-cell absolute CN, row-aligned to the h5ad (NaN
  where a cell has no paired scDNA). For the Section-2.2 per-cell benchmark, restrict to the 390
  tumor cells (`obs['cn_role'] == 'tumor'`).

Tier-1 ground truth was called from the paired single-cell DNA reads with BWA-MEM + SAMtools +
AneuFinder (1 Mb bins, e-divisive segmentation) and projected to gene level (A375/HCT116);
scONE-seq provides paired scWGS from the source study.

## Tier 2  -  bulk-CN cell-line panels (per-line DepMap ground truth)
Pooled panels of distinct cell lines; each cell is scored against the **bulk** DepMap copy number of
its line (`obs['cell_line']`), so these measure per-line generalization rather than per-cell DNA.

### `ccle_esophageal/`  -  7 CCLE esophageal lines (Kinker et al. 2020, GSE157220 / SCP542)
- `ccle_esophageal.h5ad`  -  2,543 cells x 7,351 genes, raw UMI, `obs['cell_line']`.
- `ccle_esophageal_cn.npy`  -  `(2543, 7351)` bulk DepMap absolute CN (24Q4), broadcast to each line's
  cells, row-aligned to the h5ad.

### `ccle_mixed/`  -  3 CCLE mixed-tissue lines (CL34, NCIH460, SKMEL5; GSE157220 / SCP542)
- `ccle_mixed.h5ad`  -  1,512 cells x 7,351 genes, raw UMI, `obs['cell_line']`.
- `ccle_mixed_cn.npy`  -  `(1512, 7351)` bulk DepMap absolute CN (24Q4), broadcast per line.

### `mixseq/`  -  MIX-seq pan-cancer panel (McFarland et al. 2020, figshare 10.6084/m9.figshare.10298696)
- `mixseq_pancancer.h5ad`  -  3,055 cells x 17,768 genes across **49 pan-cancer lines** (the
  DepMap-CN-covered subset of the frozen 73-line MIX-seq experiment), raw UMI, `obs['cell_line']`.
- `mixseq_pancancer_cn.npy`  -  `(3055, 17768)` bulk DepMap absolute CN (24Q4), broadcast per line,
  row-aligned to the h5ad (NaN where a gene is absent from DepMap).

## Scoring
`refcon.metrics.per_cell_metrics`  -  per-cell Pearson/Spearman and AUROC (loss `CN < 1.5`,
gain `CN > 2.5`) vs the absolute-CN profiles; correlation is scale-invariant, so REFCON's mean-1
ratios score directly. `per_line_metrics` aggregates the same over `obs['cell_line']` for Tier 2.

The training cohorts are **not** included in this release; see the paper's Data availability for their
primary accessions.
