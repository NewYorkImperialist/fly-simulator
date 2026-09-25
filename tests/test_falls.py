"""Fall detector + run metrics.

Physical tests use a real sim and shove the thorax by writing data.xfrc_applied
(stand-in for the perturbation module). The recovery path cannot be produced
physically (the baseline controller never rights itself once on its back), so it
is exercised with a scripted fake sim.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from perpetualfly import AppConfig, Simulation
from perpetualfly.metrics import (
    ContactSummary,
    FallDetector,
    FallDetectorConfig,
    FallState,
    RunMetrics,
)


def shove(sim: Simulation, direction, multiple_of_weight: float, duration: float):
    """Constant world-frame force on the thorax COM for ``duration`` seconds."""
    t0 = sim.time
    f = np.asarray(direction, float) * multiple_of_weight * sim.fly_mass * 9810.0  # uN

    def hook(s):
        s.data.xfrc_applied[s.thorax_body_id, :3] = f if s.time < t0 + duration else 0.0

    sim.pre_step_hooks.append(hook)
    return {"time": t0, "direction": list(direction),
            "magnitude": float(np.linalg.norm(f)), "duration": duration}, hook


@pytest.fixture(scope="module")
def rig():
    sim = Simulation(AppConfig())
    det = FallDetector(sim)
    met = RunMetrics().attach(sim, det)
    events = []
    det.add_listener(events.append)
    yield sim, det, met, events
    sim.close()


def test_no_false_falls_normal_walking(rig):
    sim, det, met, events = rig
    sim.step(120_000)  # 12 s
    kinds = [e.kind for e in events]
    assert kinds == [], f"false detections: {[(e.kind, e.time, e.reason) for e in events]}"
    assert det.state == FallState.UPRIGHT
    assert met.n_falls == 0 and met.longest_jog_interval > 11.5
    assert met.distance > 140  # ~14 mm/s


def test_strong_shove_falls_and_reset_is_upright(rig):
    sim, det, met, events = rig
    sim.reset()
    assert [e.kind for e in events][-1:] == ["reset"]
    events.clear()
    sim.step(20_000)  # 2 s of walking
    hit, hook = shove(sim, [0, -1, 0], 10.0, 0.010)  # 10x body weight, 10 ms, rightwards
    met.record_hit(hit)
    try:
        sim.step(15_000)
    finally:
        sim.pre_step_hooks.remove(hook)
    kinds = [e.kind for e in events]
    assert "fall" in kinds, kinds
    assert det.state == FallState.FALLEN
    assert met.n_falls == 1 and met.n_hits == 1 and met.hits[0].caused_fall
    assert met.current_jog_interval == 0.0
    fall = next(e for e in events if e.kind == "fall")
    assert fall.tilt_deg > 60

    sim.reset()
    assert det.state == FallState.UPRIGHT
    assert events[-1].kind == "reset"
    assert met.n_resets == 2 and met.state == FallState.UPRIGHT
    sim.step(5000)
    assert det.state == FallState.UPRIGHT  # no spurious event right after reset
    assert met.summary()["n_falls"] == 1  # counts survive resets


# ---------------------------------------------------------------- scripted fake sim
class FakeSim:
    """Just enough of Simulation for FallDetector.update()."""

    def __init__(self):
        self.thorax_body_id = 0
        self.time = 0.0
        self.step_count = 0
        self.data = SimpleNamespace(xpos=np.zeros((1, 3)), xmat=np.zeros((1, 9)))
        self.set(0.0, 1.15, 0.0, body=False)

    def set(self, x, h, tilt_deg, body):
        a = np.radians(tilt_deg)  # roll about x
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        self.data.xpos[0] = [x, 0.0, h]
        self.data.xmat[0] = R.ravel()
        self.body = body

    def tilt_deg(self):
        return float(np.degrees(np.arccos(self.data.xmat[0, 8])))


class FakeClassifier:
    def __init__(self, sim):
        self.sim = sim

    def summarize(self, data):
        b = self.sim.body
        return ContactSummary(3, 0 if b else 6, np.full(6, not b), b, int(b))


def run_script(det, sim, seconds, speed, h, tilt, body, dt=1e-3):
    for _ in range(int(round(seconds / dt))):
        sim.time += dt
        x = sim.data.xpos[0, 0] + speed * dt
        sim.set(x, h, tilt, body)
        det.update(sim)


def test_state_machine_fall_and_recovery():
    sim = FakeSim()
    det = FallDetector(cfg=FallDetectorConfig())
    det._classifier = FakeClassifier(sim)
    met = RunMetrics()
    det.add_listener(met.on_fall_event)
    kinds = []
    det.add_listener(lambda e: kinds.append(e.kind))
    met.update(0.0, np.zeros(3))

    run_script(det, sim, 3.0, 14.0, 1.15, 5.0, False)  # walking
    assert kinds == [] and det.state == FallState.UPRIGHT
    run_script(det, sim, 0.05, 50.0, 1.4, 70.0, False)  # tipping
    run_script(det, sim, 1.0, 0.0, 0.55, 160.0, True)  # on its back
    assert kinds == ["destabilized", "fall"] and det.state == FallState.FALLEN
    run_script(det, sim, 0.5, 14.0, 1.15, 5.0, False)  # rights itself, walks
    assert det.state == FallState.RECOVERING
    run_script(det, sim, 1.5, 14.0, 1.15, 5.0, False)
    assert kinds[-2:] == ["recovering", "recovered"] and det.state == FallState.UPRIGHT
    assert met.n_falls == 1 and met.n_recoveries == 1
    assert 1.0 < met.recovery_times[0] < 2.0
    assert met.recovery_percentage == 100.0

    # Mild disturbance: DESTABILIZED then back to UPRIGHT, no fall.
    kinds.clear()
    run_script(det, sim, 0.05, 14.0, 0.8, 20.0, False)
    run_script(det, sim, 0.6, 14.0, 1.15, 5.0, False)
    assert kinds == ["destabilized", "stabilized"]


def test_airborne_tumble_is_not_a_fall():
    sim = FakeSim()
    det = FallDetector()
    det._classifier = FakeClassifier(sim)
    kinds = []
    det.add_listener(lambda e: kinds.append(e.kind))
    run_script(det, sim, 2.0, 14.0, 1.15, 5.0, False)
    run_script(det, sim, 0.6, 30.0, 8.0, 150.0, False)  # tumbling high in the air
    run_script(det, sim, 1.0, 14.0, 1.15, 5.0, False)  # lands on its feet
    assert kinds == ["destabilized", "stabilized"]


def test_body_contact_alone_is_not_a_fall_but_stuck_is():
    sim = FakeSim()
    det = FallDetector()
    det._classifier = FakeClassifier(sim)
    kinds = []
    det.add_listener(lambda e: kinds.append(e.kind))
    run_script(det, sim, 2.0, 14.0, 1.15, 5.0, False)
    run_script(det, sim, 0.1, 14.0, 1.0, 5.0, True)  # abdomen scrapes a rock briefly
    run_script(det, sim, 1.0, 14.0, 1.15, 5.0, False)
    assert kinds == ["destabilized", "stabilized"]
    run_script(det, sim, 3.0, 0.0, 0.9, 10.0, True)  # belly on the ground, not moving
    assert kinds[-1] == "fall" and det.state == FallState.FALLEN


def test_run_metrics_hits_and_eval_stats():
    met = RunMetrics(hit_fall_window_s=2.0)
    met.update(0.0, np.zeros(3))
    met.update(10.0, np.array([140.0, 0, 0]))
    h = met.record_hit({"time": 9.0, "direction": "left", "magnitude": 35.0, "duration": 0.01})
    assert h.impulse == pytest.approx(0.35) and met.n_hits == 1
    s = met.summary()
    assert s["n_hits_survived"] == 1 and s["max_force_survived_uN"] == 35.0
    assert s["mean_time_between_failures_s"] is None and s["falls_per_km"] == 0.0
    assert met.longest_jog_interval == pytest.approx(10.0)
    assert "falls=0 rec=0 hits=1" in met.summary_line()
