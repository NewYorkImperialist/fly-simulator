"""Wingbeat pattern generator (per-wing stroke / deviation / rotation angles).

Phase ``p`` (rad) advances at ``2*pi*freq``; at p = 0 both wings are at the
forward (ventral) stroke reversal.

* stroke     phi(p)   = phi0 + A/2 * cos(p)            (+ = forward)
  p in (0, pi): upstroke (wing moves backward); p in (pi, 2pi): downstroke
  (forward, the main lift stroke in hovering *Drosophila*).
* rotation   psi(p): leading-edge-first at geometric angle of attack
  ``aoa_down`` on the downstroke (psi = aoa_down) and ``aoa_up`` on the upstroke
  (psi = pi - aoa_up: the wing is flipped, leading edge pointing backward). The
  flip is a smoothed square wave ``tanh(k sin(p + rot_phase)) / tanh(k)``;
  ``rot_phase > 0`` rotates the wing *before* stroke reversal (advanced rotation).
* deviation  theta(p) = -atan(tan(tilt) * sin(phi - phi0)) : the wing tip leaves
  the nominal stroke plane so that the effective stroke plane is rotated by
  ``tilt`` about the lateral axis (+ = front of the stroke lower = nose-down tilt,
  force vector tilted forward). The same mechanism with opposite signs on the two
  wings would roll the plane; not used.

Per-side parameters allow left/right asymmetry (roll: amplitude, yaw: angle of
attack / rotation timing). All angles in rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

DEG = math.pi / 180.0


@dataclass
class WingParams:
    amplitude: float = 151.0 * DEG  # peak-to-peak stroke
    mean_stroke: float = 0.0  # phi0, + = forward (moves the centre of lift forward)
    aoa_down: float = 45.0 * DEG
    aoa_up: float = 45.0 * DEG
    rot_phase: float = 0.0  # rotation advance (rad of wingbeat phase)
    tilt: float = 0.0  # stroke-plane tilt via deviation, + = nose-down (forward force)


@dataclass
class WingbeatParams:
    freq: float = 218.0  # Hz
    left: WingParams = field(default_factory=WingParams)
    right: WingParams = field(default_factory=WingParams)
    rot_sharpness: float = 3.0  # k of the tanh flip (large = square wave)

    def symmetric(self, **kw) -> "WingbeatParams":
        """Copy with the same WingParams overrides on both wings."""
        return replace(self, left=replace(self.left, **kw), right=replace(self.right, **kw))


def wing_angles(p: float, w: WingParams, k: float) -> tuple[float, float, float]:
    """(stroke, deviation, rotation) of one wing at phase p."""
    half = 0.5 * w.amplitude
    dphi = half * math.cos(p)
    phi = w.mean_stroke + dphi
    theta = -math.atan(math.tan(w.tilt) * math.sin(dphi)) if w.tilt else 0.0
    s = math.tanh(k * math.sin(p + w.rot_phase)) / math.tanh(k)
    lo, hi = w.aoa_down, math.pi - w.aoa_up  # downstroke / upstroke rotation angles
    psi = 0.5 * (lo + hi) + 0.5 * (hi - lo) * s
    return phi, theta, psi


class WingbeatGenerator:
    """Integrates the wingbeat phase (so frequency changes stay continuous) and
    returns the 6 wing joint targets ``[l_stroke, l_dev, l_rot, r_stroke, r_dev,
    r_rot]`` plus their time derivatives (finite difference over the phase).

    ``envelope`` (0..1) scales the pattern toward the spread, flat rest pose
    (all angles 0): ``start(ramp_s)`` fades the wingbeat in from rest, ``stop``
    fades it out. Starting at full amplitude from rest demands ~1e8 rad/s^2 on the
    light rotation hinge and destabilises the (explicit) fluid forces.
    """

    def __init__(self, params: WingbeatParams | None = None) -> None:
        self.params = params or WingbeatParams()
        self.phase = 0.0
        self.envelope = 1.0
        self._env_rate = 0.0  # 1/s, + fading in, - fading out

    def reset(self, phase: float = 0.0) -> None:
        self.phase = phase
        self.envelope = 1.0
        self._env_rate = 0.0

    def start(self, ramp_s: float = 0.015, phase: float = 0.0) -> None:
        self.phase = phase
        self.envelope = 0.0 if ramp_s > 0 else 1.0
        self._env_rate = 1.0 / ramp_s if ramp_s > 0 else 0.0

    def stop(self, ramp_s: float = 0.015) -> None:
        self._env_rate = -1.0 / max(ramp_s, 1e-9)

    def angles(self, phase: float | None = None) -> np.ndarray:
        p = self.phase if phase is None else phase
        P = self.params
        return np.array(wing_angles(p, P.left, P.rot_sharpness)
                        + wing_angles(p, P.right, P.rot_sharpness))

    def targets(self, phase: float | None = None, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        p = self.phase if phase is None else phase
        q = self.angles(p)
        dq = (self.angles(p + eps) - q) / eps * (2.0 * math.pi * self.params.freq)
        e = self.envelope
        if e < 1.0:
            # smoothstep envelope and its time derivative
            sm = e * e * (3.0 - 2.0 * e)
            dsm = 6.0 * e * (1.0 - e) * self._env_rate
            return sm * q, sm * dq + dsm * q
        return q, dq

    def advance(self, dt: float) -> None:
        self.phase += 2.0 * math.pi * self.params.freq * dt
        if self._env_rate:
            self.envelope = min(1.0, max(0.0, self.envelope + self._env_rate * dt))
            if self.envelope in (0.0, 1.0):
                self._env_rate = 0.0

    @property
    def stopped(self) -> bool:
        return self.envelope == 0.0 and self._env_rate == 0.0

    @property
    def cycle(self) -> int:
        return int(self.phase // (2.0 * math.pi))
