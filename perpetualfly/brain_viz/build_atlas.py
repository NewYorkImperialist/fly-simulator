"""Build ``assets/flywire_neuropils_frontal.json`` from real FlyWire neuropil meshes.

One-off developer tool (the generated JSON is committed; the window never runs this).

Sources (downloaded as plain wheels, *not* installed -- no navis/trimesh needed):

* ``fafbseg`` wheel -> ``fafbseg/data/JFRC2NP.surf.fw.zip``: the 78 standard
  neuropils (Ito et al. 2014, JFRC2 template) transformed into FlyWire (FAFB14.1)
  space, one PLY per neuropil, units nm. These are the same neuropil names FlyWire
  uses to annotate synapses (``AL_L``, ``MB_CA_R``, ``GNG``, ...).
* ``flybrains`` wheel -> ``flybrains/meshes/FLYWIRE_whole_brain.ply``: brain surface
  (with optic lobes) in the same space, used for the silhouette.

Projection: straight down the anterior-posterior (z) axis, the usual FlyWire/Codex
view: ``u = x / 1000``, ``v = y / 1000`` in micrometres. In this space the ``*_L``
neuropils have the smaller x, so the fly's LEFT is on the image LEFT (as if seen from
behind the fly, like the fly window's chase camera) and y grows ventrally (image
down). z (anterior -> posterior) is kept per region as ``depth``.

Usage::

    python -m perpetualfly.brain_viz.build_atlas --fafbseg-whl fafbseg.whl \
        --flybrains-whl flybrains.whl [--out perpetualfly/brain_viz/assets/...json]
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from perpetualfly.brain_viz.atlas import ATLAS_PATH, region_group

PX_UM = 2.0  # raster resolution used to compute silhouettes (um per pixel)


def read_ply(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary-little-endian PLY reader: float xyz vertices + triangle faces."""
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end:end + 2] == b"\r\n" else 1  # exactly one line break
    header = data[:end].decode("ascii").splitlines()
    if not any("binary_little_endian" in h for h in header):
        raise ValueError("only binary_little_endian PLY supported")
    n_v = n_f = 0
    vprops: list[str] = []
    count_t = index_t = "i4"
    types = {"uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4", "uint": "u4",
             "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8"}
    cur = None
    for h in header:
        p = h.split()
        if p[:2] == ["element", "vertex"]:
            n_v, cur = int(p[2]), "v"
        elif p[:2] == ["element", "face"]:
            n_f, cur = int(p[2]), "f"
        elif p[:1] == ["property"] and cur == "v":
            vprops.append(types[p[1]])
        elif p[:2] == ["property", "list"] and cur == "f":
            count_t, index_t = types[p[2]], types[p[3]]
    vdt = np.dtype([(f"p{i}", "<" + t) for i, t in enumerate(vprops)])
    body = data[end:]
    v = np.frombuffer(body, vdt, n_v)
    verts = np.stack([v["p0"], v["p1"], v["p2"]], 1).astype(np.float64)
    fdt = np.dtype([("n", "<" + count_t), ("i", "<" + index_t, 3)])
    f = np.frombuffer(body, fdt, n_f, offset=vdt.itemsize * n_v)
    if not np.all(f["n"] == 3):
        raise ValueError("non-triangle faces")
    return verts, f["i"].astype(np.int64)


def silhouette(verts_um: np.ndarray, faces: np.ndarray, origin: np.ndarray,
               shape: tuple[int, int], min_area_px: float = 12.0):
    """Rasterise the projected triangles; return (outlines_um, centroid_um, area_um2)."""
    mask = np.zeros(shape, np.uint8)
    pts = np.round((verts_um[:, :2] - origin) / PX_UM * 4).astype(np.int32)  # 2 bit subpx
    cv2.fillPoly(mask, list(pts[faces]), 255, lineType=cv2.LINE_8, shift=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outlines = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area_px:
            continue
        c = cv2.approxPolyDP(c, 0.8, True)[:, 0, :].astype(np.float64)
        outlines.append(np.round(c * PX_UM + origin, 1).tolist())
    ys, xs = np.nonzero(mask)
    centroid = (np.array([xs.mean(), ys.mean()]) * PX_UM + origin) if len(xs) else \
        verts_um[:, :2].mean(0)
    return outlines, np.round(centroid, 1).tolist(), float(len(xs) * PX_UM ** 2)


def build(fafbseg_whl: Path, flybrains_whl: Path, out: Path) -> dict:
    with zipfile.ZipFile(flybrains_whl) as z:
        bverts, bfaces = read_ply(z.read("flybrains/meshes/FLYWIRE_whole_brain.ply"))
    with zipfile.ZipFile(fafbseg_whl) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read("fafbseg/data/JFRC2NP.surf.fw.zip")))
    names = sorted(n[:-4] for n in inner.namelist()
                   if n.endswith(".ply") and "/" not in n and not n.startswith("."))
    meshes = {n: read_ply(inner.read(f"{n}.ply")) for n in names}

    bverts = bverts / 1000.0
    lo = bverts[:, :2].min(0) - 20
    hi = bverts[:, :2].max(0) + 20
    for v, _ in meshes.values():
        lo = np.minimum(lo, v[:, :2].min(0) / 1000 - 20)
        hi = np.maximum(hi, v[:, :2].max(0) / 1000 + 20)
    shape = (int((hi[1] - lo[1]) / PX_UM) + 1, int((hi[0] - lo[0]) / PX_UM) + 1)

    brain_outline, _, _ = silhouette(bverts, bfaces, lo, shape, min_area_px=200)
    regions = {}
    for n, (v, f) in meshes.items():
        v = v / 1000.0
        outlines, centroid, area = silhouette(v, f, lo, shape)
        regions[n] = {
            "centroid": centroid,
            "depth": round(float(v[:, 2].mean()), 1),
            "area_um2": round(area),
            "group": region_group(n),
            "outline": outlines,
        }
    atlas = {
        "format": "perpetualfly-neuropil-atlas-v1",
        "source": "fafbseg JFRC2NP.surf.fw (Ito 2014 neuropils in FlyWire space) + "
                  "flybrains FLYWIRE_whole_brain.ply",
        "units": "micrometres, FlyWire/FAFB14.1 space (nm / 1000)",
        "projection": "along z: u = x_nm/1000 (*_L neuropils at smaller u = image "
                      "left), v = y_nm/1000 (grows ventrally = image down); "
                      "depth = mean z um",
        "bounds": [round(float(lo[0]), 1), round(float(lo[1]), 1),
                   round(float(hi[0]), 1), round(float(hi[1]), 1)],
        "brain_outline": brain_outline,
        "regions": regions,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, separators=(",", ":")))
    return atlas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fafbseg-whl", type=Path, required=True)
    ap.add_argument("--flybrains-whl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ATLAS_PATH)
    a = ap.parse_args()
    atlas = build(a.fafbseg_whl, a.flybrains_whl, a.out)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} KiB, "
          f"{len(atlas['regions'])} regions)")


if __name__ == "__main__":
    main()
