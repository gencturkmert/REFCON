#!/usr/bin/env python3
"""Train a single REFCON member (Conv-binned per-bin ratio model) at one context length.

Usage (invoked per member by scripts/train.py):
    python -m refcon.train_binned_dev --train-data <dir> --val-data <dir> \
        --bin-size 8 --max-chunk 1024 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# refcon/refcon/train_binned_dev.py -> repo root is 1 level up (refcon/).
# Set REFCON_ROOT to redirect training output/logs elsewhere.
import os
PROJECT_ROOT = Path(os.environ.get("REFCON_ROOT", str(Path(__file__).resolve().parents[1])))

from .model_binned_dev import BinnedDevModel
from .data import load_dataloaders

# Training run outputs (checkpoints) and logs, repo-relative by default.
BASE_DIR = PROJECT_ROOT / "runs"
LOG_DIR = PROJECT_ROOT / "runs" / "logs"
DEFAULTS = {
    "seed": 42,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "dim_ff": 512,
    "dropout": 0.1,
    "conv_kernel": 11,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "early_stop_patience": 30,
    "var_target_std": 0.15,
    "var_penalty_weight": 0.1,
}


def bin_ratio_targets(cn, lengths, bin_size):
    """Compute per-bin ratio targets.

    Returns (bin_ratios, bin_valid, chunk_mean).
    """
    B, S = cn.shape
    n_bins = S // bin_size
    mask = torch.arange(S, device=cn.device).unsqueeze(0) < lengths.unsqueeze(1)
    valid = (cn > -0.5) & mask
    safe_cn = cn.clone()
    safe_cn[~valid] = 0.0
    counts = valid.float().sum(dim=1).clamp(min=1)
    chunk_mean = safe_cn.sum(dim=1) / counts

    ratios = safe_cn / chunk_mean.unsqueeze(1).clamp(min=0.1)
    ratios[~valid] = 0.0

    usable = n_bins * bin_size
    ratios_trim = ratios[:, :usable].view(B, n_bins, bin_size)
    valid_trim = valid[:, :usable].view(B, n_bins, bin_size)
    bin_count = valid_trim.float().sum(dim=2)
    bin_sum = (ratios_trim * valid_trim.float()).sum(dim=2)
    bin_ratios = bin_sum / bin_count.clamp(min=1)
    bin_valid = bin_count >= 3

    return bin_ratios, bin_valid, chunk_mean


@torch.no_grad()
def evaluate_binned(model, datasets, device, max_samples=2000):
    """Evaluate BinnedDevModel on a set of datasets: bin-level AND gene-level metrics.

    ``datasets`` is the dict returned by ``load_dataloaders`` ({name: {"ds", "loader"}}).
    """
    model.eval()
    all_metrics = []

    for ds_name, ds_dict in datasets.items():
        dataset = ds_dict["ds"]
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

        all_bin_pred, all_bin_gt = [], []
        all_gene_pred, all_gene_gt = [], []
        n = 0

        for expr, cn, length in loader:
            if n >= max_samples:
                break
            expr = expr.to(device)
            cn = cn.float()
            length_d = length.to(device)

            bin_pred, bin_lengths = model(expr, length_d)
            bin_pred = bin_pred.cpu()
            bin_gt, bin_valid, _ = bin_ratio_targets(cn, length, model.bin_size)

            gene_pred = model.predict_per_gene(expr, length_d).cpu()
            B, S = cn.shape
            gene_mask = torch.arange(S).unsqueeze(0) < length.unsqueeze(1)
            gene_valid = (cn > -0.5) & gene_mask

            # Per-gene ratio GT
            safe = cn.clone()
            safe[~gene_valid] = 0.0
            counts = gene_valid.float().sum(dim=1).clamp(min=1)
            cmean = safe.sum(dim=1) / counts
            gene_ratio_gt = cn / cmean.unsqueeze(1).clamp(min=0.1)

            for i in range(cn.shape[0]):
                if n >= max_samples:
                    break
                # Bin level
                bv = bin_valid[i]
                if bv.sum() >= 3:
                    all_bin_pred.append(bin_pred[i][bv])
                    all_bin_gt.append(bin_gt[i][bv])
                # Gene level
                gv = gene_valid[i]
                if gv.sum() >= 5:
                    all_gene_pred.append(gene_pred[i][gv])
                    all_gene_gt.append(gene_ratio_gt[i][gv])
                n += 1

        if not all_bin_pred:
            continue

        bp = torch.cat(all_bin_pred).double()
        bg = torch.cat(all_bin_gt).double()
        gp = torch.cat(all_gene_pred).double()
        gg = torch.cat(all_gene_gt).double()

        def pearson(a, b):
            if a.std() < 1e-8 or b.std() < 1e-8:
                return 0.0
            cov = ((a - a.mean()) * (b - b.mean())).mean()
            return float(cov / (a.std() * b.std()))

        def spearman(a, b):
            if len(a) < 10:
                return 0.0
            ra = a.argsort().argsort().double()
            rb = b.argsort().argsort().double()
            return pearson(ra, rb)

        def macro_f1(pred, gt, lo=0.75, hi=1.25):
            pc = torch.ones_like(pred, dtype=torch.long)
            pc[pred < lo] = 0; pc[pred > hi] = 2
            gc = torch.ones_like(gt, dtype=torch.long)
            gc[gt < lo] = 0; gc[gt > hi] = 2
            f1s = []
            for c in (0, 1, 2):
                tp = ((pc == c) & (gc == c)).sum().float()
                fp = ((pc == c) & (gc != c)).sum().float()
                fn = ((pc != c) & (gc == c)).sum().float()
                p = (tp / (tp + fp).clamp(min=1e-8)).item()
                r = (tp / (tp + fn).clamp(min=1e-8)).item()
                f1s.append(2 * p * r / max(p + r, 1e-8))
            return float(np.mean(f1s))

        m = {
            "_ds_name": ds_name,
            "_n_samples": n,
            "bin_pearson": pearson(bp, bg),
            "bin_spearman": spearman(bp, bg),
            "bin_mae": float((bp - bg).abs().mean()),
            "bin_macro_f1": macro_f1(bp, bg),
            "gene_pearson": pearson(gp, gg),
            "gene_spearman": spearman(gp, gg),
            "gene_mae": float((gp - gg).abs().mean()),
            "gene_macro_f1": macro_f1(gp, gg),
            # Backward compat aliases
            "pearson": pearson(bp, bg),
            "spearman": spearman(bp, bg),
            "macro_f1": macro_f1(bp, bg),
            "dev_ratio_spearman": spearman(bp, bg),
        }
        all_metrics.append(m)

    return all_metrics


def train(cfg, device, train_data, val_data):
    bs = cfg["bin_size"]
    mc = cfg["max_chunk"]
    run_id = f"bdev_bs{bs}_mc{mc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = BASE_DIR / "checkpoints" / "dev" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.log"

    log = logging.getLogger(run_id)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for h in [logging.FileHandler(log_path), logging.StreamHandler()]:
        h.setFormatter(fmt)
        log.addHandler(h)

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log.info("Run: %s", run_id)
    log.info("=" * 70)
    log.info("BINNED DEV - bin_size=%d, max_chunk=%d, n_bins=%d", bs, mc, mc // bs)
    log.info("=" * 70)
    for k, v in sorted(cfg.items()):
        log.info("  %-25s = %s", k, v)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    log.info("Loading data (norm_mode=raw, max_chunk=%d)...", mc)
    train_sets = load_dataloaders(
        train_data, norm_mode="raw", max_chunk=mc, batch_size=cfg["batch_size"],
        loss_oversample=cfg.get("loss_oversample", 2.0),
        coverage=cfg.get("coverage", "random"), shuffle=True,
    )
    val_sets = load_dataloaders(
        val_data, norm_mode="raw", max_chunk=mc, batch_size=cfg["batch_size"],
        loss_oversample=1.0, coverage="random", shuffle=False,
    )

    train_loaders = {name: d["loader"] for name, d in train_sets.items()}
    log.info("Training: %s | validation: %s",
             list(train_loaders.keys()), list(val_sets.keys()))

    model = BinnedDevModel(
        bin_size=bs, max_chunk=mc,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
    ).to(device)
    log.info("Model: %s params, n_bins=%d", f"{model.count_parameters():,}", model.n_bins)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.get("lr_patience", 5),
        factor=cfg.get("lr_factor", 0.5), min_lr=cfg.get("lr_min", 1e-6),
    )

    best_composite = -float("inf")
    best_epoch = -1
    patience = 0

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        iterators = {n: iter(l) for n, l in train_loaders.items()}
        totals = {"loss": 0.0, "ratio": 0.0, "var": 0.0}
        bc = 0
        finished = set()
        pstds = []

        while len(finished) < len(train_loaders):
            for name in list(train_loaders.keys()):
                if name in finished:
                    continue
                try:
                    expr, cn, length = next(iterators[name])
                except StopIteration:
                    finished.add(name)
                    continue

                expr = expr.to(device)
                cn = cn.to(device).float()
                length = length.to(device)

                bin_pred, bin_lengths = model(expr, length)
                bin_gt, bin_valid, _ = bin_ratio_targets(cn, length, bs)
                if not bin_valid.any():
                    continue

                L_ratio = F.smooth_l1_loss(bin_pred[bin_valid], bin_gt[bin_valid])
                pred_std = bin_pred[bin_valid].std()
                L_var = F.relu(torch.tensor(cfg.get("var_target_std", 0.15), device=device) - pred_std)
                loss = L_ratio + cfg.get("var_penalty_weight", 0.1) * L_var

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                optimizer.step()

                totals["loss"] += loss.item()
                totals["ratio"] += L_ratio.item()
                totals["var"] += L_var.item()
                pstds.append(pred_std.item())
                bc += 1

        elapsed = time.time() - t0
        avg = {k: v / max(bc, 1) for k, v in totals.items()}
        lr = optimizer.param_groups[0]["lr"]
        log.info("")
        log.info("Epoch %3d/%d [%.1fs]", epoch + 1, cfg["max_epochs"], elapsed)
        log.info("TRAIN: loss=%.4f (ratio=%.4f var=%.4f) pred_std=%.4f lr=%.1e",
                avg["loss"], avg["ratio"], avg["var"],
                sum(pstds) / len(pstds) if pstds else 0, lr)

        # Validation
        merged = evaluate_binned(model, val_sets, device)
        composite = np.mean([m.get("bin_spearman", 0) for m in merged]) if merged else 0.0
        log.info("VAL [bin+gene level]:")
        for m in merged:
            log.info("  %-22s bin_pear=%.4f bin_spear=%.4f bin_mf1=%.4f | gene_pear=%.4f gene_mf1=%.4f",
                     m["_ds_name"], m["bin_pearson"], m["bin_spearman"], m["bin_macro_f1"],
                     m["gene_pearson"], m["gene_macro_f1"])
        log.info("  composite (bin_spearman): %.4f  patience %d/%d",
                composite, patience, cfg.get("early_stop_patience", 30))

        scheduler.step(composite)

        if composite > best_composite:
            best_composite = composite
            best_epoch = epoch + 1
            patience = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "composite": composite, "config": cfg,
            }, ckpt_dir / "best.pt")
            log.info("  >> New best: %.4f @ epoch %d", composite, epoch + 1)
        else:
            patience += 1

        if patience >= cfg.get("early_stop_patience", 30):
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    log.info("=" * 70)
    log.info("DONE | best=%.4f @ epoch %d | stopped @ epoch %d",
             best_composite, best_epoch, epoch + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-data", required=True, help="directory (or file) of processed training .h5ad(s)")
    ap.add_argument("--val-data", required=True, help="directory (or file) of processed validation .h5ad(s)")
    ap.add_argument("--bin-size", type=int, required=True)
    ap.add_argument("--max-chunk", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--coverage", default="random")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--early-stop-patience", type=int, default=30)
    ap.add_argument("--var-target-std", type=float, default=None)
    ap.add_argument("--var-penalty-weight", type=float, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update({
        "bin_size": args.bin_size,
        "max_chunk": args.max_chunk,
        "max_epochs": args.epochs,
        "coverage": args.coverage,
        "lr": args.lr,
        "d_model": args.d_model,
        "n_heads": max(1, args.d_model // 32),
        "dim_ff": args.d_model * 4,
        "batch_size": args.batch_size,
        "early_stop_patience": args.early_stop_patience,
    })
    if args.var_target_std is not None:
        cfg["var_target_std"] = args.var_target_std
    if args.var_penalty_weight is not None:
        cfg["var_penalty_weight"] = args.var_penalty_weight

    train(cfg, torch.device(args.device), args.train_data, args.val_data)


if __name__ == "__main__":
    main()
