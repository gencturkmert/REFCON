# Expected input format

REFCON reads a single **AnnData `.h5ad`** per sample.

## Expression matrix `adata.X`
- **Raw UMI counts** (integers), shape `(n_cells, n_genes)`. Do **not** pre-normalize  -  a per-cell
  masked InstanceNorm inside the model removes library-size and platform scale.
- Dense or sparse both work.
- Smart-seq2 / plate data that only ship TPM-derived values are accepted as-is (the model never
  computes TPM); pass whatever expression you have in `X`.

## Gene metadata `adata.var` (required topological features)
Genes must be **sorted by genomic position within each chromosome**, and `var` must carry:

| column        | meaning                                                            |
|---------------|-------------------------------------------------------------------|
| `chromosome`  | chromosome label per gene (e.g. `chr1`…`chr22`)                   |
| `start`,`end` | gene genomic coordinates                                          |
| `abspos`      | absolute genome position (used for ordering)                     |
| `chr_boundary`| `1` for the first gene of each chromosome, else `0`              |

REFCON is **gene-identity-agnostic**: it uses only these positions and chromosome boundaries, never
gene names or gene-set memberships. A single trained model therefore applies across gene panels and
platforms (10x, Smart-seq2, DNTR-seq, scONE-seq, BD Rhapsody) without retraining.

Example (`data/a375/a375_dntr_per_cell.h5ad`):
```
adata.X          int32 raw UMI, (96, 12506)
adata.var cols   gene_id, chromosome, start, end, abspos, bp_gap, chr_boundary
```

## Output
`scripts/infer.py` writes `ens3_cn.npz` with:
- `cn`     -  `(n_cells, n_genes)` per-gene **copy-number ratio**, renormalized so each cell's mean over
  valid genes is **1** (relative dosage, not absolute copy number / ploidy).
- `cells`  -  cell ids (= `adata.obs_names`)
- `genes`  -  gene symbols (= `adata.var_names`)

## Ground-truth convention (for scoring)
Per-cell **absolute** copy number, one value per gene. A gene/cell with no reliable call is marked
`NaN` **or** a negative sentinel (`-1`) and is excluded from scoring (valid = finite and `>= 0`)  - 
including the `-1` no-calls as if they were copy number would under-report Pearson. Metrics:
per-cell Pearson/Spearman, and AUROC with **loss = GT < 1.5**, **gain = GT > 2.5**. Correlation is
scale-invariant, so REFCON's mean-1 ratio scores directly against absolute-CN ground truth.
