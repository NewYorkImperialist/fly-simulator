"""The game's eyes: looming of each rock on each compound eye -> LC4 / LPLC2 events.

``GameVision`` is ``fly_simulator.vision.looming.LoomingVision`` (same field of view,
response function type, event format, brain mapping and event scheduling) with two
game-interface changes, both motivated by artefacts measured with walking flies:

1. **Gaze stabilisation.** The model's eyes are rigid on the thorax, and the tripod
   gait yaws the thorax by about +-7 deg at 12 Hz. Eye frames here follow the body
   yaw low-passed with ``gaze_tau_s`` (80 ms) and stay level with the ground. Real
   flies stabilise gaze against body oscillations with head movements (Kim et al.
   2017, Nat Neurosci 20:1096; Cruz et al. 2021, Curr Biol 31:4596).
2. **Expansion is measured on the object, visibility gates it.** The base class
   takes the angular size from the field-of-view-*masked* solid angle, so an object
   sliding across the soft edge of an eye's field of view changes "size" and reads
   as expansion or contraction. With the rocks (which pass through the frontal
   binocular edge on their way in) this gave the *contralateral* eye up to ~200 Hz
   of spurious LC4 drive, which would steer the fly toward the rock. Here each eye
   computes the rock's true angular size theta = 2 asin(r / distance) (spheres),
   low-passes it (``LoomingConfig.tau_s``), differentiates it, applies the source's
   ``LoomResponse`` and multiplies both rates by the eye's field-of-view weight in
   the rock's direction (``looming.fov_mask``: 1 inside, 0 outside, smooth over
   ``fov_edge_deg``). LC4 / LPLC2 are expansion detectors; an object moving out of
   their receptive field is not looming.

Everything else (``_maybe_send``: events of ``persist_s``, refreshed, re-sent on a
rise; ``StimulusEvent("loom", side=eye, details={lc4_hz, lplc2_hz, ...})``) is the
base class's.
"""

from __future__ import annotations

import math

import numpy as np

from fly_simulator.vision.looming import EYES, EyeLoom, LoomingVision, fov_mask, loom_response


class GameVision(LoomingVision):
    def __init__(self, *a, gaze_tau_s: float = 0.08, **kw) -> None:
        super().__init__(*a, **kw)
        self.gaze_tau_s = float(gaze_tau_s)
        self._yaw: float | None = None
        self._yaw_t: float | None = None

    # ------------------------------------------------------------------ gaze
    def _on_reset(self, sim) -> None:
        super()._on_reset(sim)
        self._yaw = self._yaw_t = None

    def gaze_yaw(self) -> float:
        t, h = self.sim.time, self.sim.heading()
        if self._yaw is None or self._yaw_t is None or t < self._yaw_t:
            self._yaw = h
        else:
            a = 1.0 - math.exp(-(t - self._yaw_t) / max(self.gaze_tau_s, 1e-6))
            self._yaw += a * ((h - self._yaw + math.pi) % (2 * math.pi) - math.pi)
        self._yaw_t = t
        return self._yaw

    def eyes(self):
        E, _ = super().eyes()
        y = self.gaze_yaw()
        c, s = math.cos(y), math.sin(y)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return E, np.stack([Rz, Rz])

    # ------------------------------------------------------------------ looming
    def update(self):
        cfg = self.cfg
        t = self.sim.time
        dt = None if self._last_t is None else t - self._last_t
        if dt is not None and (dt <= 0 or dt > 0.1):
            dt = None
        self._last_t = t
        E, R = self.eyes()
        best = [None, None]
        for src in self.sources:
            c, _ax, _half, r = self.shapes(src)
            resp = src.response or cfg.response
            if len(r) == 0:
                th = np.zeros(2)
                w = np.zeros(2)
                az = np.zeros(2)
                el = np.zeros(2)
                dist = np.full(2, math.inf)
            else:
                D = c[0][None, :] - E  # (2, 3)
                rho = np.maximum(np.linalg.norm(D, axis=1), 1e-9)
                th = 2.0 * np.arcsin(np.clip(r[0] / rho, 0.0, 1.0))
                h = np.einsum("nji,nj->ni", R, D / rho[:, None])  # head-frame directions
                w = fov_mask(h, self.side_sign, cfg)
                az = np.degrees(np.arctan2(h[:, 1] * self.side_sign, h[:, 0]))
                el = np.degrees(np.arcsin(np.clip(h[:, 2], -1.0, 1.0)))
                dist = rho - r[0]
            f = self._filt.get(src.name)
            if f is None or dt is None or len(r) == 0:
                f = np.array(th, dtype=float)
                dth = np.zeros(2)
            else:
                a = 1.0 - math.exp(-dt / max(cfg.tau_s, 1e-9))
                new = f + a * (th - f)
                dth = (new - f) / dt
                f = new
            self._filt[src.name] = f
            st = self.state[src.name]
            for i in range(2):
                lc4, lplc2 = loom_response(math.degrees(f[i]), math.degrees(dth[i]), resp)
                lc4 *= float(w[i])
                lplc2 *= float(w[i])
                lc4 = lc4 if lc4 >= resp.min_hz else 0.0
                lplc2 = lplc2 if lplc2 >= resp.min_hz else 0.0
                e = EyeLoom(theta=math.degrees(f[i]), theta_raw=math.degrees(th[i]),
                            dtheta=math.degrees(dth[i]), az=float(az[i]), el=float(el[i]),
                            dist=float(dist[i]), lc4_hz=lc4, lplc2_hz=lplc2)
                st[i] = e
                if len(r):
                    self.history.append((t, src.name, EYES[i], e))
                score = max(lc4, lplc2)
                if best[i] is None or score > best[i][0]:
                    best[i] = (score, src.name, e)
        self.n_updates += 1
        for i in range(2):
            if best[i] is not None:
                self._maybe_send(i, t, best[i][1], best[i][2])
        return self.state

    def eye_weights(self, src_name: str) -> tuple[float, float]:
        """Current (left, right) LC4 drive of one source (diagnostics / HUD)."""
        st = self.state.get(src_name)
        return (st[0].lc4_hz, st[1].lc4_hz) if st else (0.0, 0.0)
