#!/usr/bin/env python3
"""Fetch REFCON's released data + checkpoints from Zenodo into ./data and ./checkpoints.

The processed evaluation cohorts (expression + copy-number ground truth) and the three model
checkpoints are hosted on Zenodo as two archives, data.zip and checkpoints.zip (they are too
large to keep in git). Run this once after cloning.

  python scripts/fetch_data.py
"""
import urllib.request, zipfile, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZENODO_RECORD = "10.5281/zenodo.21644534"

# archive on Zenodo -> top-level folder it extracts into (relative to repo root)
ARCHIVES = {
    "data.zip":        "data",         # data/<cohort>/... (one subdirectory per evaluation cohort)
    "checkpoints.zip": "checkpoints",  # the three ensemble-member .pt checkpoints
}


def main():
    base = f"https://zenodo.org/records/{ZENODO_RECORD.split('.')[-1]}/files"
    for fn, dst in ARCHIVES.items():
        target = ROOT / dst
        if target.exists() and any(target.iterdir()):
            print(f"[skip ] {dst}/ already populated")
            continue
        url = f"{base}/{fn}?download=1"
        print(f"[get  ] {url}")
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        try:
            urllib.request.urlretrieve(url, tmp.name)
            print(f"[unzip] {fn} -> {dst}/")
            with zipfile.ZipFile(tmp.name) as z:
                z.extractall(ROOT)
        finally:
            os.unlink(tmp.name)
    print("done.")


if __name__ == "__main__":
    main()
