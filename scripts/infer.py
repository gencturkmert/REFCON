#!/usr/bin/env python3
"""REFCON ensemble inference (ens3 = context lengths 512 + 1024 + 2048).

Runs the three released checkpoints on a processed .h5ad of RAW UMI counts, averages the per-cell
copy-number ratios (this mean is the REFCON result reported in the paper), and records runtime and
peak memory.

Usage
  python scripts/infer.py --data <sample>.h5ad --out out/sample --device cuda:0

Outputs (in --out): ens3_cn.npz (cn, cells, genes) and runtime.json.
Reference-free by design: each cell is predicted independently; no reference panel is used.
"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent           # refcon/
CKPTS = [("mc512", "ensemble_bs8_mc512.pt"),             # short context
         ("mc1024", "baseline_bs8_mc1024.pt"),          # medium context
         ("mc2048", "ensemble_bs8_mc2048.pt")]          # long context


def run_checkpoint(ckpt, h5ad, out_pt, device, cell_batch):
    # Run eval_cellwide as a package module so its relative imports resolve; put the repo root on
    # PYTHONPATH so this works from a fresh clone (no install) and after `pip install -e .` alike.
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "refcon.eval_cellwide",
           "--checkpoint", str(ROOT / "checkpoints" / ckpt),
           "--data", str(h5ad), "--output", str(out_pt),
           "--device", device, "--cell-batch", str(cell_batch)]
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    return rc, time.time() - t0


def load_pt(p):
    import torch
    d = torch.load(p, map_location="cpu", weights_only=False)
    return (np.asarray(d["predicted_cn_ratios"], float),
            np.asarray(d["cell_ids"]).astype(str),
            np.asarray(d["gene_symbols"]).astype(str))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="processed raw-UMI .h5ad (var has chromosome/chr_boundary)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda:0", help="cuda / mps / cpu (falls back to cpu on failure)")
    ap.add_argument("--cell-batch", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    preds, cells, genes, timing, dev_used = [], None, None, {}, a.device
    for tag, ck in CKPTS:
        o = out / f"pred_{tag}.pt"
        rc, dt = run_checkpoint(ck, a.data, o, a.device, a.cell_batch)
        if rc != 0 and a.device != "cpu":
            print(f"[{tag}] {a.device} failed; retrying on cpu", flush=True)
            rc, dt = run_checkpoint(ck, a.data, o, "cpu", a.cell_batch); dev_used = "cpu"
        if rc != 0:
            sys.exit(f"[{tag}] inference FAILED (rc={rc})")
        p, cells, genes = load_pt(o)
        preds.append(p); timing[tag] = round(dt, 2)
        print(f"[{tag}] {p.shape} in {dt:.1f}s", flush=True)

    ens = np.mean(preds, 0).astype(np.float32)
    np.savez(out / "ens3_cn.npz", cn=ens, cells=cells, genes=genes)

    total = sum(timing.values())
    # ru_maxrss: bytes on macOS, kilobytes on Linux
    unit = 1e6 if sys.platform == "darwin" else 1e3
    runtime = dict(device=dev_used, n_cells=int(ens.shape[0]), n_genes=int(ens.shape[1]),
                   n_checkpoints=3, cell_batch=a.cell_batch,
                   per_checkpoint_s=timing, total_inference_s=round(total, 2),
                   cells_per_s=round(ens.shape[0] / max(total, 1e-9), 2),
                   peak_rss_mb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / unit, 1))
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print("RUNTIME", json.dumps(runtime))


if __name__ == "__main__":
    main()
