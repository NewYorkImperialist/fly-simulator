import csv
import json

import numpy as np
import pytest

from perpetualfly import AppConfig, Simulation
from perpetualfly.metrics import FallDetector, LoggingConfig, RunLogger, RunMetrics
from perpetualfly.metrics.run_logger import EVENT_COLUMNS, METRIC_COLUMNS, quat_to_rpy_deg


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_logger_files_rate_and_summary(tmp_path):
    cfg = AppConfig()
    sim = Simulation(cfg)
    det = FallDetector(sim)
    met = RunMetrics().attach(sim, det)
    log_cfg = LoggingConfig(runs_dir=str(tmp_path), sample_hz=100.0)
    logger = RunLogger(log_cfg, config={"app": cfg, "logging": log_cfg},
                       terrain_type_fn=lambda x: "flat").attach(sim, det, met)
    try:
        # Interim state is usable before close (crash safety): headers flushed.
        assert read_csv(logger.run_dir / "metrics.csv")[0] == METRIC_COLUMNS
        sim.step(20_000)  # 2 s
        t0 = sim.time
        f = np.array([0.0, -10.0 * sim.fly_mass * 9810.0, 0.0])  # 10x weight, rightwards

        def hook(s):
            s.data.xfrc_applied[s.thorax_body_id, :3] = f if s.time < t0 + 0.01 else 0.0

        sim.pre_step_hooks.append(hook)
        met.record_hit(time=t0, direction=[0, -1, 0], magnitude=float(np.linalg.norm(f)),
                       duration=0.01)
        sim.step(10_000)
        met.record_hit(direction="left", magnitude=5.0, duration=0.01)
    finally:
        logger.close()
        sim.close()
    logger.close()  # idempotent

    d = logger.run_dir
    assert d.parent == tmp_path
    assert sorted(p.name for p in d.iterdir()) == [
        "config.json", "events.csv", "metrics.csv", "summary.json"]
    config = json.loads((d / "config.json").read_text())
    assert config["app"]["controller"]["kind"] == "hybrid"
    assert AppConfig.from_dict(config["app"]) == cfg

    rows = read_csv(d / "metrics.csv")
    assert rows[0] == METRIC_COLUMNS
    data = rows[1:]
    t = np.array([float(r[0]) for r in data])
    assert np.allclose(np.diff(t), 0.01, atol=1e-6)  # 100 Hz of sim time
    assert len(data) == pytest.approx(3.0 * 100 + 1, abs=2)
    assert all(len(r) == len(METRIC_COLUMNS) for r in data)
    col = {name: i for i, name in enumerate(METRIC_COLUMNS)}
    assert data[-1][col["state"]] == "FALLEN" and data[-1][col["terrain_type"]] == "flat"
    assert float(data[-1][col["tilt_deg"]]) > 60

    ev = read_csv(d / "events.csv")
    assert ev[0] == EVENT_COLUMNS
    types = [r[EVENT_COLUMNS.index("event_type")] for r in ev[1:]]
    summary = json.loads((d / "summary.json").read_text())
    m = summary["metrics"]
    assert summary["complete"] is True
    assert m["n_falls"] == types.count("fall") == 1
    assert m["n_hits"] == types.count("hit") == 2
    assert m["n_recoveries"] == types.count("recovered") == 0
    assert m["n_hits_survived"] == 1  # the second one came after the fall
    assert summary["n_metric_samples"] == len(data)
    fall_row = next(r for r in ev[1:] if r[EVENT_COLUMNS.index("event_type")] == "fall")
    assert fall_row[EVENT_COLUMNS.index("fall_detected")] == "1"
    hit_row = next(r for r in ev[1:] if r[EVENT_COLUMNS.index("event_type")] == "hit")
    assert float(hit_row[EVENT_COLUMNS.index("force_magnitude")]) > 90
    json.loads(fall_row[EVENT_COLUMNS.index("details")])


def test_rpy_convention():
    yaw, pitch, roll = np.radians([30.0, -10.0, 5.0])
    cz, sz, cy, sy, cx, sx = (np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch),
                              np.cos(roll), np.sin(roll))
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    assert quat_to_rpy_deg(Rz @ Ry @ Rx) == pytest.approx((5.0, -10.0, 30.0))


def test_run_dir_collision(tmp_path):
    a = RunLogger(LoggingConfig(runs_dir=str(tmp_path)), config={}, run_name="x")
    b = RunLogger(LoggingConfig(runs_dir=str(tmp_path)), config={}, run_name="x")
    a.close()
    b.close()
    assert a.run_dir != b.run_dir and b.run_dir.name == "x_1"
    assert json.loads((b.run_dir / "summary.json").read_text())["complete"] is True


def test_absolute_columns_keep_precision_in_long_runs(tmp_path):
    # Regression: with 5 significant digits a 30 min run logged timestamps at 0.1 s
    # resolution (50 Hz rows collided) and x ~ 25 m at 1 mm resolution.
    logger = RunLogger(LoggingConfig(runs_dir=str(tmp_path)), config={})
    try:
        logger.log_event("spawn", 12345.6789, position=(25000.12345, -3.25, 1.5),
                         force_magnitude=123456.789)
    finally:
        logger.close()
    row = dict(zip(EVENT_COLUMNS, read_csv(logger.run_dir / "events.csv")[1]))
    assert float(row["timestamp"]) == pytest.approx(12345.6789, abs=1e-4)
    assert float(row["sim_time"]) == pytest.approx(12345.6789, abs=1e-4)
    assert float(row["x"]) == pytest.approx(25000.12345, abs=1e-4)
    assert row["force_magnitude"] == "1.2346e+05"  # relative columns keep %.5g
