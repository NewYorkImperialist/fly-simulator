"""The delivery pilot job (fly_simulator/jobs/delivery_pilot.py): a job on the real
flight fly (``needs_flight``: flapping wings, air, dt 5e-5 s) and the app / runner
wiring that allows it; the town (colliding roofs, visual decoration, a mocap parcel
pool); one real delivery hop (take-off from the depot with the Jump -> wings, climb,
cruise, approach, FlightMode landing on house 1, the parcel carried under the fly and
dropped on the doormat); the counters and HUD; a missed landing with a retry; and the
explicit, counted recovery (the parcel goes back to the shelf, the order stays open)."""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.app import ConfigError, apply_feature_defaults, check_feature_config
from fly_simulator.config import AppConfig, RenderConfig
from fly_simulator.flight import FLIGHT_TIMESTEP, FlightSimulation
from fly_simulator.jobs import available_jobs, create_job_session, get_job, make_job
from fly_simulator.jobs.delivery_pilot import P, DeliveryPilotJob


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _run(session, seconds: float, until=None) -> None:
    t_end = session.run_time() + seconds
    while session.run_time() < t_end:
        session.sim.step(100)
        session.after_physics()
        if until is not None and until():
            return


@pytest.fixture(scope="module")
def hop():
    """One delivery to house 1 (~2.1 s of sim), with a trace of the carried parcel."""
    msgs: list[str] = []
    session, job = create_job_session("delivery_pilot", _cfg(), {"wind_max": 0.0},
                                      say=msgs.append)
    trace = {"phases": [], "carry_dz": [], "max_z": 0.0}
    sim = session.sim
    t_end = session.run_time() + 3.2
    while session.run_time() < t_end and job.n_delivered == 0:
        sim.step(100)
        session.after_physics()
        if not trace["phases"] or trace["phases"][-1] != job.phase:
            trace["phases"].append(job.phase)
        com = sim.com()
        trace["max_z"] = max(trace["max_z"], float(com[2]))
        if job.phase == "cruise" and job.carrying is not None:
            p = sim.data.mocap_pos[job.mocap_parcel[job.carrying]]
            trace["carry_dz"].append((float(p[2] - com[2]), float(np.hypot(*(p[:2] - com[:2])))))
    yield session, job, trace, msgs
    sim.close()


def test_registered_needs_flight_and_other_jobs_unchanged():
    assert "delivery_pilot" in available_jobs()
    assert get_job("delivery_pilot") is DeliveryPilotJob
    assert DeliveryPilotJob.needs_flight
    for name in available_jobs():
        if name not in ("delivery_pilot", "crop_duster"):  # the flying jobs
            assert not get_job(name).needs_flight, name
    # a walking job keeps the canonical walking model
    cfg = AppConfig()
    make_job("sisyphus").configure_app(cfg)
    assert not cfg.flight.enabled and cfg.sim.timestep == AppConfig().sim.timestep
    # the flying job switches the flight fly on
    cfg = AppConfig()
    job = make_job("delivery_pilot")
    job.configure_app(cfg)
    assert cfg.flight.enabled and cfg.sim.timestep == FLIGHT_TIMESTEP
    assert cfg.render.render_every_steps == 2 * RenderConfig().render_every_steps
    assert cfg.flight.hover_s is None and not cfg.fly.extra_joints
    job.configure_app(cfg)  # idempotent (the app path calls it after apply_feature_defaults)
    assert cfg.render.render_every_steps == 2 * RenderConfig().render_every_steps


def test_app_allows_flight_only_for_flying_jobs():
    cfg = AppConfig()
    cfg.job.name, cfg.flight.enabled = "delivery_pilot", True
    check_feature_config(cfg)  # --flight --job delivery_pilot: fine
    cfg = AppConfig()
    cfg.job.name = "delivery_pilot"
    apply_feature_defaults(cfg)  # --job delivery_pilot alone turns the flight fly on
    assert cfg.flight.enabled and cfg.sim.timestep == FLIGHT_TIMESTEP
    cfg = AppConfig()
    cfg.job.name, cfg.flight.enabled = "sisyphus", True
    with pytest.raises(ConfigError):
        check_feature_config(cfg)
    cfg = AppConfig()
    cfg.course.name, cfg.flight.enabled = "gauntlet", True
    with pytest.raises(ConfigError):
        check_feature_config(cfg)


def test_scene_on_the_flight_fly(hop):
    session, job, _, _ = hop
    sim, m = session.sim, session.sim.model
    assert isinstance(sim, FlightSimulation) and sim.timestep == FLIGHT_TIMESTEP
    assert session.flight is not None and m.opt.density > 0
    # roofs / walls / attics collide like terrain; decoration and parcels are visual
    for n in (P + "depot_roof", P + "house1_roof", P + "house1_walls", P + "house1_attic"):
        g = m.geom(n)
        assert g.contype[0] != 0 and g.conaffinity[0] != 0, n
    for n in (P + "house1_doormat", P + "parcel0_box", P + "windsock", P + "depot_sign"):
        gids = ([m.geom(n).id] if mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n) >= 0
                else [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(n).id])
        for g in gids:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, n
    assert all(m.body_mocapid[m.body(P + f"parcel{k}").id] >= 0 for k in range(job.n_parcels))
    # the walkable surface: roofs, street
    assert job.ground_height(0.0, 0.0) == pytest.approx(job.cfg.depot_top)
    x, y, top, _ = job.houses[0]
    assert job.ground_height(x, y) == pytest.approx(top)
    assert job.ground_height(20.0, -60.0) == 0.0


def test_one_delivery_hop_real_flight(hop):
    session, job, trace, msgs = hop
    fm = session.flight
    assert job.n_delivered == 1 and job.per_house[0] == 1
    # the state machine: load -> take-off (Jump -> wings) -> climb -> cruise ->
    # approach -> FlightMode landing -> landed -> drop
    ph = trace["phases"]
    for a, b in zip(("load", "takeoff", "climb", "cruise", "approach", "landing", "landed"),
                    ("takeoff", "climb", "cruise", "approach", "landing", "landed", "drop")):
        assert ph.index(a) < ph.index(b)
    assert fm.counts["takeoff"] == 1 and fm.counts["land"] == 1 and fm.counts["crash"] == 0
    assert trace["max_z"] > job.cfg.cruise_alt - 1.0  # climbed to cruise altitude
    assert job.on_pad(0) and session.sim.tilt_deg() < 20.0
    assert job.n_falls == 0 and job.n_missed == 0 and job.n_crash == 0
    # the parcel hung under the fly while cruising (kinematic, straight below the COM)
    assert trace["carry_dz"]
    dz = np.array(trace["carry_dz"])
    assert np.allclose(dz[:, 0], job.cfg.carry_dz, atol=0.05) and dz[:, 1].max() < 1e-6
    # ... and was dropped on the doormat of house 1
    k = next(i for i, st in enumerate(job.parcel_state) if st == ("mat", 0))
    assert job.carrying is None
    _run(session, 0.5)  # the drop slide finishes (and the next take-off starts)
    p = session.sim.data.mocap_pos[job.mocap_parcel[k]]
    x, y, top, _ = job.houses[0]
    assert np.hypot(p[0] - x, p[1] - y) < job.cfg.house_half and p[2] > top
    assert any("DELIVERED #1" in m for m in msgs)
    # counters
    st = job.stats()
    assert st["delivered"] == 1 and st["on_time_pct"] == 100.0
    assert 0.8 < st["flight_time_s"] < 3.0 and 38.0 < st["distance_mm"] < 70.0
    assert job.work == 1


def test_hud_caption_and_camera(hop):
    session, job, _, _ = hop
    lines = job.hud_lines()
    text = "\n".join(lines)
    assert "DELIVERY PILOT" in lines[0] and "parcels delivered: 1" in text
    assert "flight time" in text and "on time" in text and "crash landings" in text
    assert "kinematically" in text  # the parcel carry is labelled
    job._flash("DELIVERED!", "No. 1 *ding-dong*")
    frame = np.zeros((106, 160, 3), np.uint8)
    out = job.post_process(frame.copy(), job.run_time())
    assert out.any()  # the caption is drawn
    assert not job.post_process(frame.copy(), job.run_time() + 10.0).any()
    pre = job.camera_preset()
    assert math.isfinite(pre.azimuth) and pre.distance == job.cfg.cam_distance


def test_missed_landing_retry_and_recovery():
    session, job = create_job_session("delivery_pilot", _cfg(), {"wind_max": 0.0},
                                      say=lambda m: None)
    try:
        sim = session.sim
        _run(session, 0.3)
        assert job.phase == "load" and job.carrying is not None  # the parcel is loaded
        k = job.carrying
        # a landing judged while the fly stands on the depot but was flying to house 2:
        # a missed landing -> counted, then a retry take-off from where it is
        job.dest = 1
        job._set("landed")
        job._judge_landing()
        assert job.n_missed == 1 and job.phase == "wait"
        _run(session, 1.2, until=lambda: job.phase == "takeoff")
        assert job.phase == "takeoff" and job.n_takeoffs == 1
        _run(session, 0.2, until=lambda: job.phase == "climb")
        assert session.flight.airborne
        # a fall -> the explicit, counted recovery: respawn on the depot roof, the
        # parcel back on the shelf, the order stays open
        order = job.order
        job.recover("fell")
        assert job.n_auto_recoveries == 1 and job.recovery_reasons == {"fell": 1}
        assert job.phase == "load" and job.carrying is None and job.parcel_state[k] == "shelf"
        assert job.next_house == order and not session.flight.airborne
        assert np.allclose(sim.data.mocap_pos[job.mocap_parcel[k]], job._shelf_slot(k))
        assert sim.model.opt.wind[:2].tolist() == [0.0, 0.0]
        _run(session, 0.3)
        assert job.carrying is not None and job.order == order  # reloaded for the same address
    finally:
        session.sim.close()


def test_wind_is_the_medium_velocity():
    job = make_job("delivery_pilot", {"wind_max": 20.0})
    session, job = create_job_session(job, _cfg(), say=lambda m: None)
    try:
        _run(session, 0.05)
        w = session.sim.model.opt.wind[:2]
        assert 0.0 < float(np.hypot(*w)) <= 20.0 + 1e-9
        assert np.allclose(w, job._wind)
    finally:
        session.sim.close()
