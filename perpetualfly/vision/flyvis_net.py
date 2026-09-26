"""flyvis connectome-constrained visual system, stepped frame by frame (docs/VISION.md).

"Real vision" part 2. flyvis (Lappalainen et al. 2024, Nature 634:1132; TuragaLab/
flyvis 1.2) is a model of the fly's optic lobe (retina, lamina, medulla, lobula
plate: 65 cell types, 45 669 neurons per eye on a 721-column hexagonal lattice) whose
connectivity is fixed from the connectome and whose single-neuron / synapse
parameters were trained on optic-flow estimation. It ends at T4 / T5 (the ON / OFF
elementary motion detectors, 4 direction subtypes a-d each) and their lobula-plate
partners; it does NOT contain LC4 / LPLC2 (see ``bridge.py``).

``flyvis.Network.simulate`` integrates a whole movie per call (~1 s overhead), so a
closed loop needs a stepwise integrator. ``StepwiseFlyvis`` ports FlyGym v1's
``RealTimeVisionNetwork`` idea (NeLy-EPFL/flygym-gymnasium, examples/vision/
vision_network.py) without subclassing: keep the network state and call the
network's own Euler step (``Network._next_state``) once per ommatidia frame, with
``dt = 1 / eyes rate``. Both eyes are one batch of 2.

``RetinaMapper`` is a port of FlyGym v1's class of the same name: FlyGym's and
flyvis's hex lattices have the same 721 ommatidia but different orderings; a linear
map between the extreme ommatidia of both lattices gives the permutation.

Needs the optional ``vision`` extra (``pip install -e ".[vision]"``; flyvis + its
~230 MB of dependencies) and the pretrained models (``flyvis download-pretrained
--skip_large_files``, ~10 MB). ``flyvis_available()`` checks both.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass

import numpy as np

T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")
INSTALL_HINT = ('.venv/bin/python -m pip install -e ".[vision]" && '
                ".venv/bin/flyvis download-pretrained --skip_large_files")


def flyvis_available() -> str | None:
    """None if flyvis and a pretrained model are available, else the reason."""
    if importlib.util.find_spec("flyvis") is None or importlib.util.find_spec("torch") is None:
        return f"flyvis / torch not installed ({INSTALL_HINT})"
    try:
        import flyvis
    except Exception as e:  # broken install
        return f"flyvis import failed: {e}"
    d = flyvis.results_dir / "flow/0000/000"
    if not (d / "best_chkpt").exists() and not any(d.glob("chkpts/*")):
        return f"flyvis pretrained models missing in {d} ({INSTALL_HINT})"
    return None


class RetinaMapper:
    """Permutation between FlyGym's ``Retina`` ommatidia order and flyvis's ``BoxEye``
    hexal order (port of FlyGym v1's ``RetinaMapper``). Also gives each flyvis hexal's
    centre in FlyGym eye-image pixels (``centers_rc``: row, col)."""

    def __init__(self) -> None:
        from flygym.vision.retina import Retina
        from flyvis.datasets.rendering import BoxEye

        retina = Retina()
        rc = BoxEye(extent=15).receptor_centers.cpu().numpy().astype(float)
        cols = np.unique(rc[:, 1])
        left = rc[rc[:, 1] == cols[0]]
        right = rc[rc[:, 1] == cols[-1]]
        fv_a = np.array([left[:, 0].min(), cols[0]])
        fv_b = np.array([right[:, 0].max(), cols[-1]])
        idm = retina.ommatidia_id_map
        ar, ac = np.where(idm == 1)
        br, bc = np.where(idm == idm.max())
        fg_a = np.array([ar.mean(), ac.mean()])
        fg_b = np.array([br.mean(), bc.mean()])
        k = (fg_b - fg_a) / (fv_b - fv_a)
        b = fg_a - k * fv_a
        c = rc * k + b  # flyvis centres in FlyGym image coordinates
        self.centers_rc = c
        ri = np.clip(np.round(c).astype(int), 0, np.array(idm.shape) - 1)
        self.idx_flyvis_to_flygym = idm[ri[:, 0], ri[:, 1]].astype(np.int64) - 1
        if np.any(self.idx_flyvis_to_flygym < 0) or len(np.unique(self.idx_flyvis_to_flygym)) != len(c):
            raise RuntimeError("RetinaMapper: flyvis / FlyGym hex lattices do not line up")
        self.idx_flygym_to_flyvis = np.argsort(self.idx_flyvis_to_flygym)

    def flygym_to_flyvis(self, x: np.ndarray) -> np.ndarray:
        return x[..., self.idx_flyvis_to_flygym]

    def flyvis_to_flygym(self, x: np.ndarray) -> np.ndarray:
        return x[..., self.idx_flygym_to_flyvis]


@dataclass
class FlyvisConfig:
    model: str = "flow/0000/000"  # pretrained ensemble member (results_dir-relative)
    fade_in_s: float = 0.5  # contrast fade-in of the first frame (network warm-up)
    threads: int = 2  # torch intra-op threads (the brain + physics need the rest)


class StepwiseFlyvis:
    """The flyvis network, integrated one frame at a time for both eyes.

    ``step(frame)``: frame (2, 721) luminance in FlyGym ommatidia order (left, right);
    returns the node activity (2, n_nodes) (torch tensor). ``motion()`` gives the
    latest T4 / T5 activity as (2, 4, 721) arrays per family in *FlyGym* order.
    """

    def __init__(self, dt: float, cfg: FlyvisConfig | None = None) -> None:
        import torch
        import flyvis
        from flyvis.network import NetworkView

        self.cfg = cfg or FlyvisConfig()
        self.dt = float(dt)
        if self.cfg.threads:
            torch.set_num_threads(int(self.cfg.threads))
        logging.getLogger("flyvis").setLevel(logging.WARNING)
        t0 = time.perf_counter()
        self.net = NetworkView(flyvis.results_dir / self.cfg.model).init_network()
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.init_s = time.perf_counter() - t0
        self._torch = torch
        self.mapper = RetinaMapper()
        with torch.no_grad():
            self.net.clamp()
            self.params = self.net._param_api()
        li = self.net.connectome.nodes.layer_index
        self.idx_t4 = np.stack([np.asarray(li[t]) for t in T4_TYPES])  # (4, 721)
        self.idx_t5 = np.stack([np.asarray(li[t]) for t in T5_TYPES])
        # flyvis orders every columnar type like the input hexals (checked below)
        nodes = self.net.connectome.nodes
        u, v = nodes.u[:], nodes.v[:]
        r1 = np.asarray(li["R1"])
        if not all(np.array_equal(u[r1], u[i]) and np.array_equal(v[r1], v[i])
                   for i in (*self.idx_t4, *self.idx_t5)):
            raise RuntimeError("flyvis: T4 columns are not in input-hexal order")
        self.state = None
        self.activity = None
        self.n_steps = 0
        self.wall_s = 0.0

    # -------------------------------------------------------------- core
    def _x_t(self, frame_fv):
        torch = self._torch
        net = self.net
        x = torch.as_tensor(np.ascontiguousarray(frame_fv), dtype=torch.float32)
        net.stimulus.zero(x.shape[0], 1)
        net.stimulus.add_input(x[:, None, None, :])
        return net.stimulus()[:, 0]

    def reset(self, frame: np.ndarray) -> None:
        """Warm the network up on ``frame`` (contrast fade-in, like FlyGym v1)."""
        torch = self._torch
        fv = self.mapper.flygym_to_flyvis(np.asarray(frame, dtype=np.float32))
        with torch.no_grad():
            init = torch.as_tensor(fv, dtype=torch.float32)[:, None, :]
            self.state = self.net.fade_in_state(self.cfg.fade_in_s, self.dt, init)
            self.activity = self.state.nodes.activity

    def step(self, frame: np.ndarray):
        t0 = time.perf_counter()
        if self.state is None:
            self.reset(frame)
        fv = self.mapper.flygym_to_flyvis(np.asarray(frame, dtype=np.float32))
        with self._torch.no_grad():
            self.state = self.net._next_state(self.params, self.state, self._x_t(fv), self.dt)
            self.activity = self.state.nodes.activity
        self.n_steps += 1
        self.wall_s += time.perf_counter() - t0
        return self.activity

    def ms_per_step(self) -> float:
        return 1e3 * self.wall_s / max(self.n_steps, 1)

    # -------------------------------------------------------------- readouts
    def cell_activity(self, idx: np.ndarray) -> np.ndarray:
        """Activity of node indices ``idx`` (k, 721) in FlyGym order -> (2, k, 721)."""
        a = self.activity.cpu().numpy()
        out = a[:, idx]  # (2, k, 721) flyvis hexal order
        return out[..., self.mapper.idx_flygym_to_flyvis]

    def motion(self) -> tuple[np.ndarray, np.ndarray]:
        """(T4, T5): each (2 eyes, 4 subtypes a-d, 721) in FlyGym ommatidia order."""
        return self.cell_activity(self.idx_t4), self.cell_activity(self.idx_t5)
