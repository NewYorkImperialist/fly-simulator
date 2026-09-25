"""Loading the FlyWire v783 connectome + annotations used by the brain model.

Files (downloaded by ``scripts/fetch_brain_data.py`` into ``data/brain/``):

* ``Completeness_783.csv`` + ``Connectivity_783.parquet``: Shiu et al. 2024 model
  inputs (github.com/philshiu/Drosophila_brain_model, MIT code; FlyWire data).
  Row order of the completeness table defines the model's neuron index.
* ``flywire_neuron_annotations.tsv``: Schlegel et al. 2024 / Berg et al. 2025
  annotations (github.com/flyconnectome/flywire_annotations, v3.1.0): super class,
  cell type, top predicted neurotransmitter (Eckstein et al. 2024), side, positions.
* ``per_neuron_neuropil_count_pre_783.feather``: presynapse counts per neuron and
  neuropil (Dorkenwald et al. 2024, Zenodo 10.5281/zenodo.10676866); used to give
  each neuron a primary (output) neuropil.

``build_neuron_table()`` condenses the annotations into ``neurons_783.npz`` (a few MB,
model order) so the brain process does not need to parse the 30 MB TSV at start-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import NEUROTRANSMITTERS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "brain"

COMPLETENESS = "Completeness_783.csv"
CONNECTIVITY = "Connectivity_783.parquet"
ANNOTATIONS = "flywire_neuron_annotations.tsv"
NEUROPIL_PRE = "per_neuron_neuropil_count_pre_783.feather"
NEURON_TABLE = "neurons_783.npz"

NT_CODE = {"acetylcholine": 0, "gaba": 1, "glutamate": 2, "dopamine": 3,
           "serotonin": 4, "octopamine": 5}

# FlyWire voxel size (nm) for pos_* / soma_* columns.
VOXEL_NM = np.array([4.0, 4.0, 40.0])

STR_COLS = ("super_class", "cell_class", "cell_sub_class", "cell_type",
            "hemibrain_type", "side", "flow")


def data_available(data_dir: Path | str | None = None) -> bool:
    d = Path(data_dir or DEFAULT_DATA_DIR)
    return all((d / f).is_file() for f in (COMPLETENESS, CONNECTIVITY, NEURON_TABLE))


def load_root_ids(data_dir: Path | str | None = None) -> np.ndarray:
    import pyarrow.csv as pc

    d = Path(data_dir or DEFAULT_DATA_DIR)
    t = pc.read_csv(d / COMPLETENESS)
    return t.column(0).to_numpy().astype(np.int64)


def load_connectome(data_dir: Path | str | None = None):
    """Return (indptr, indices, signed_counts, n) in model order (CSR by pre)."""
    import pyarrow.parquet as pq

    from .engine import csr_from_edges

    d = Path(data_dir or DEFAULT_DATA_DIR)
    t = pq.read_table(d / CONNECTIVITY, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = t.column(0).to_numpy()
    post = t.column(1).to_numpy()
    w = t.column(2).to_numpy()
    n = len(load_root_ids(d))
    indptr, indices, weights = csr_from_edges(pre, post, w, n)
    return indptr, indices, weights, n


@dataclass
class NeuronTable:
    """Per-neuron annotations in model order."""

    root_id: np.ndarray        # (N,) int64
    nt: np.ndarray             # (N,) int8 index into NEUROTRANSMITTERS, -1 unknown
    sign: np.ndarray           # (N,) int8 +1/-1 as used by the model (0: no outputs)
    region: np.ndarray         # (N,) int16 index into ``regions``
    regions: list[str]
    pos_um: np.ndarray         # (N, 3) float32 anchor point (backbone), um; NaN unknown
    soma_um: np.ndarray        # (N, 3) float32 soma, um; NaN unknown
    cols: dict[str, np.ndarray]  # string annotation columns (object arrays)

    @property
    def n(self) -> int:
        return len(self.root_id)

    def col(self, name: str) -> np.ndarray:
        return self.cols[name]

    def index_of(self, root_ids) -> np.ndarray:
        """Model indices of FlyWire root ids (ids not in the model are dropped)."""
        lut = {int(r): i for i, r in enumerate(self.root_id)}
        return np.array([lut[int(r)] for r in root_ids if int(r) in lut], dtype=np.int64)

    def select(self, **criteria) -> np.ndarray:
        """Indices where every column equals (or is in) the given value(s)."""
        mask = np.ones(self.n, dtype=bool)
        for k, want in criteria.items():
            col = self.cols[k]
            if isinstance(want, (list, tuple, set, frozenset)):
                mask &= np.isin(col, list(want))
            else:
                mask &= col == want
        return np.nonzero(mask)[0]


def build_neuron_table(data_dir: Path | str | None = None) -> Path:
    """Condense annotations + neuropil counts into ``neurons_783.npz``."""
    import pyarrow.csv as pc
    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    d = Path(data_dir or DEFAULT_DATA_DIR)
    root = load_root_ids(d)
    n = len(root)
    lut = {int(r): i for i, r in enumerate(root)}

    ann = pc.read_csv(
        d / ANNOTATIONS,
        parse_options=pc.ParseOptions(delimiter="\t"),
        convert_options=pc.ConvertOptions(column_types={"root_id": "int64"}),
    ).to_pydict()
    rows = np.array([lut.get(int(r), -1) for r in ann["root_id"]])
    ok = rows >= 0

    out: dict[str, np.ndarray] = {"root_id": root}
    for c in STR_COLS:
        vals = np.array(["" if x is None else str(x) for x in ann[c]], dtype=object)
        full = np.full(n, "", dtype=object)
        full[rows[ok]] = vals[ok]
        cats, codes = np.unique(full.astype(str), return_inverse=True)
        out[f"{c}__cats"] = cats
        out[f"{c}__codes"] = codes.astype(np.int32)

    nt = np.full(n, -1, dtype=np.int8)
    top = [NT_CODE.get(str(x), -1) for x in ann["top_nt"]]
    nt[rows[ok]] = np.array(top, dtype=np.int8)[ok]
    out["nt"] = nt

    def xyz(prefix):
        arr = np.full((n, 3), np.nan, dtype=np.float32)
        cols = []
        for ax in "xyz":
            col = np.array([np.nan if x is None else float(x)
                            for x in ann[f"{prefix}_{ax}"]], dtype=np.float64)
            cols.append(col)
        v = np.stack(cols, axis=1) * VOXEL_NM / 1000.0
        arr[rows[ok]] = v[ok].astype(np.float32)
        return arr

    out["pos_um"] = xyz("pos")
    out["soma_um"] = xyz("soma")

    # sign as used by the model (from the Shiu connectivity table)
    t = pq.read_table(d / CONNECTIVITY, columns=["Presynaptic_Index", "Excitatory"])
    sign = np.zeros(n, dtype=np.int8)
    sign[t.column(0).to_numpy()] = t.column(1).to_numpy().astype(np.int8)
    out["sign"] = sign

    # primary output neuropil = neuropil with most presynapses
    np_t = feather.read_table(d / NEUROPIL_PRE).to_pydict()
    names = np.array(np_t["neuropil"], dtype=object)
    names = np.where(names == None, "None", names).astype(str)  # noqa: E711
    rid = np.array(np_t["pre_pt_root_id"], dtype=np.int64)
    cnt = np.array(np_t["count"], dtype=np.int64)
    idx = np.array([lut.get(int(r), -1) for r in rid])
    keep = (idx >= 0) & (names != "None")
    regions = sorted(set(names[keep]))
    rlut = {r: i for i, r in enumerate(regions)}
    best = np.full(n, -1, dtype=np.int64)
    best_cnt = np.zeros(n, dtype=np.int64)
    for i, nm, c in zip(idx[keep], names[keep], cnt[keep]):
        if c > best_cnt[i]:
            best_cnt[i] = c
            best[i] = rlut[nm]
    # neurons without presynapses in any neuropil -> "OTHER"
    if np.any(best < 0):
        regions.append("OTHER")
        best[best < 0] = len(regions) - 1
    out["region"] = best.astype(np.int16)
    out["regions"] = np.array(regions)

    path = d / NEURON_TABLE
    np.savez_compressed(path, **out)
    return path


def load_neuron_table(data_dir: Path | str | None = None) -> NeuronTable:
    d = Path(data_dir or DEFAULT_DATA_DIR)
    z = np.load(d / NEURON_TABLE, allow_pickle=False)
    cols = {}
    for c in STR_COLS:
        cats = z[f"{c}__cats"].astype(object)
        cols[c] = cats[z[f"{c}__codes"]]
    return NeuronTable(
        root_id=z["root_id"], nt=z["nt"], sign=z["sign"], region=z["region"],
        regions=[str(r) for r in z["regions"]], pos_um=z["pos_um"],
        soma_um=z["soma_um"], cols=cols)
