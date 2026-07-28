#!/usr/bin/env python3
"""Train the REFCON ensemble (the exact configuration reported in the paper).

The released model is a 3-checkpoint ensemble, one checkpoint per context length
C in {512, 1024, 2048}; the members differ ONLY in C. All other hyperparameters
are held fixed (below) and are the defaults of refcon/train_binned_dev.py.

Paper hyperparameters (Methods, "Model" and "Training"):
  architecture : 4-layer Transformer, d_model=128, 4 heads, ffn=512, dropout=0.1, pre-norm,
                 RoPE on q/k; masked InstanceNorm on log1p input; Conv1d binner (bin_size=8);
                 ~976k params / checkpoint.
  optimizer    : AdamW, lr=1e-4, weight_decay=0.01, grad-clip |.|_2=1.0, batch_size=64.
  scheduler    : ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) on the validation
                 composite metric; early stopping patience=30 (converges ~35 epochs).
  loss         : SmoothL1(beta=1) on per-bin ratio targets + 0.1*ReLU(0.15 - std(pred));
                 chromosomes carrying >=1 loss oversampled 2x.
  seed         : 42 (identical across the three members).
  input        : log1p(raw UMI counts); per-cell masked InstanceNorm handles library size/scale.

Training data is NOT bundled here (see Data availability). Point --train-data and --val-data at
directories (or single files) of processed cohorts, each a self-describing .h5ad (raw counts in X,
chr_boundary in .var, copy-number ground truth and cn_type in .uns; see docs/input_format.md).

Usage
  python scripts/train.py --train-data /path/to/train_cohorts --val-data /path/to/val_cohorts \
      --device cuda:0 --out-dir checkpoints/
  # trains the three members mc512, mc1024, mc2048.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTEXTS = [512, 1024, 2048]          # the three ensemble members
BIN_SIZE = 8
SEED = 42


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-data", required=True, help="dir (or file) of processed training .h5ad(s)")
    ap.add_argument("--val-data", required=True, help="dir (or file) of processed validation .h5ad(s)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS,
                    help="context lengths to train (default: the paper's 512 1024 2048)")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for C in a.contexts:
        cmd = [sys.executable, "-m", "refcon.train_binned_dev",
               "--bin-size", str(BIN_SIZE), "--max-chunk", str(C),
               "--train-data", a.train_data, "--val-data", a.val_data, "--device", a.device]
        print(f"\n=== training member C={C} (bin_size={BIN_SIZE}, seed={SEED}) ===\n{' '.join(cmd)}",
              flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            sys.exit(f"training C={C} failed (rc={rc})")
    print("\nDone. The paper's REFCON is the per-cell mean of the three members' predicted_cn_ratios "
          "(scripts/infer.py).")


if __name__ == "__main__":
    main()
