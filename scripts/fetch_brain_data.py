#!/usr/bin/env python
"""Download the FlyWire v783 connectome + annotations for the brain model.

    .venv/bin/python scripts/fetch_brain_data.py            # download + build table
    .venv/bin/python scripts/fetch_brain_data.py --check    # verify checksums only

Everything goes into ``data/brain/`` (gitignored). Total download ~153 MB.
URLs are pinned to commits / Zenodo records; SHA-256 values are checked after
download. Licences: Shiu et al. code MIT; FlyWire connectome data and annotations
CC BY-NC 4.0 (Zenodo record lists CC BY 4.0); see docs/BRAIN.md.

Needs pyarrow (``pip install -e .[brain]``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perpetualfly.brain.data import DEFAULT_DATA_DIR, build_neuron_table  # noqa: E402

SHIU_SHA = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"      # philshiu/Drosophila_brain_model
ANNOT_SHA = "8587524c1748ce5ef2080822a2fc890fc03bf597"     # flywire_annotations v3.1.0

FILES = [
    # (local name, url, size bytes, sha256)
    ("Completeness_783.csv",
     f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_SHA}/Completeness_783.csv",
     3_327_347, "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    ("Connectivity_783.parquet",
     f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_SHA}/Connectivity_783.parquet",
     100_804_642, "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
    ("flywire_neuron_annotations.tsv",
     f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOT_SHA}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
     31_718_505, "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"),
    ("per_neuron_neuropil_count_pre_783.feather",
     "https://zenodo.org/api/records/10676866/files/per_neuron_neuropil_count_pre_783.feather/content",
     16_853_770, "35442a46f076892dff91bd6e55fa1489b3acc64fda1d35cee7dbbbbc509a3dff"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    print(f"    downloaded in {time.time() - t0:.1f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--check", action="store_true", help="only verify existing files")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    d: Path = args.data_dir
    d.mkdir(parents=True, exist_ok=True)

    bad = 0
    for name, url, size, digest in FILES:
        p = d / name
        if p.is_file() and not args.force and sha256(p) == digest:
            print(f"ok   {name:45s} {p.stat().st_size / 1e6:7.1f} MB  sha256 {digest[:12]}")
            continue
        if args.check:
            print(f"BAD  {name} (missing or checksum mismatch)")
            bad += 1
            continue
        print(f"get  {name} ({size / 1e6:.1f} MB) <- {url}")
        download(url, p)
        got = sha256(p)
        if got != digest:
            print(f"BAD  {name}: sha256 {got} != expected {digest}")
            bad += 1
        else:
            print(f"ok   {name:45s} {p.stat().st_size / 1e6:7.1f} MB  sha256 {digest[:12]}")
    if bad:
        return 1
    if not args.check:
        t0 = time.time()
        out = build_neuron_table(d)
        print(f"built {out.name} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f} s")
    total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
    print(f"data/brain total: {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
