"""The broccoli toss job: scene, the posed viewer fly, the sequence state machine
(incl. the 0.75-1 s ponder), the snap throw (no face-on plate flip, torso twist,
chair recoil), the backward launch with a short throw-to-boom delay, the blast, the
edit effects (hit-stop / slow-mo presentation clock, post-process identity when idle,
impact flash), the room rebuild, counters, a reset mid-cycle and the (fake) brain's
bitter pulses.

One session is built for the module and runs one full cycle with shortened boom /
aftermath / rebuild phases (~9 s sim)."""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import JobRunner, available_jobs, create_job_session
from fly_simulator.jobs.broccoli_toss import BREAKABLES, P, VIEWER, BroccoliTossJob

pytestmark = pytest.mark.filterwarnings("ignore")

ORDER = ["deliver", "face", "present", "handover", "ponder", "flick", "flight", "boom",
         "aftermath", "rebuild", "walk_off", "fetch", "deliver"]


class FakeBrain:
    """Stands in for BrainLink: records what the job sends."""

    def __init__(self) -> None:
        self.sent = []
        self.stim_log = []

    def send(self, ev, source="api"):
        self.sent.append((ev, source))
        self.stim_log.append(ev)

    def update(self) -> None:
        pass


@pytest.fixture(scope="module")
def run():
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.render.width, cfg.render.height = 180, 120
    session, job = create_job_session("broccoli_toss", cfg,
                                      {"boom_s": 0.8, "aftermath_s": 0.4, "rebuild_s": 0.5, "seed": 1})
    brain = FakeBrain()
    session.brain = brain
    runner = JobRunner(session, job, headless=True)
    log = []  # (phase, run time)
    orig = job._set_phase

    def set_phase(ph, t):
        log.append((ph, t))
        orig(ph, t)

    job._set_phase = set_phase
    m, d = session.sim.model, session.sim.data
    home = np.array([d.qpos[q:q + 3].copy() for q in job.brk_q])
    max_disp = 0.0
    launch = None
    seen = [job.phase]  # the phases seen (every 15 ms chunk, deduplicated)
    fx = dict(normal_x=-1.0, twist=0.0, chair=0.0, scales=[], lag_at_boom=None, frames={})
    mount, chair = job.mount_mocap, job.chair_mocap
    while job.run_time() < 25.0:
        if job.phase == "ponder" and "idle" not in fx["frames"]:
            fr, _ = runner.render(hud=False)
            fx["frames"]["idle"] = fr
            fx["idle_same"] = job.post_process(fr, job.run_time()) is fr
        runner.step_chunk()
        if job.phase == "flick":
            fx["normal_x"] = max(fx["normal_x"], float(d.xmat[job.plate_body].reshape(3, 3)[0, 2]))
        if job.phase in ("flick", "flight", "boom"):
            q = d.mocap_quat[mount]
            fx["twist"] = max(fx["twist"], 2 * math.degrees(math.acos(min(1.0, abs(float(q[0]))))))
            fx["chair"] = max(fx["chair"], float(np.linalg.norm(d.mocap_pos[chair])))
        if job.phase == "boom":
            if fx["lag_at_boom"] is None:
                fx["lag_at_boom"] = session.edit_clock.lag
                fr, _ = runner.render(hud=False)  # the impact frame (post-processed)
                fx["frames"]["impact"] = fr
            if len(fx["scales"]) < 60:
                fx["scales"].append(session.edit_clock.last_scale)
        if job.phase != seen[-1]:
            seen.append(job.phase)
        if job.phase == "flight" and launch is None:
            launch = dict(p=job.plate_pos(), v=d.qvel[job.plate_v:job.plate_v + 3].copy())
        if job.phase in ("boom", "aftermath"):
            cur = np.array([d.qpos[q:q + 3] for q in job.brk_q])
            max_disp = max(max_disp, float(np.max(np.linalg.norm(cur - home, axis=1))))
        # one full cycle: back at "deliver" after the fetch
        if len(seen) >= len(ORDER) and seen[-1] == "deliver":
            break
    job._set_phase = orig
    fx["lag_end"] = session.edit_clock.lag
    return dict(session=session, job=job, runner=runner, log=log, seen=seen, home=home, max_disp=max_disp,
                launch=launch, brain=brain, fx=fx)


def test_registry_and_scene(run):
    assert "broccoli_toss" in available_jobs()
    job, m = run["job"], run["session"].sim.model
    assert isinstance(job, BroccoliTossJob)
    for n in (P + "plate", P + "viewer_mount", VIEWER + "/c_head", P + "blast"):
        assert mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) >= 0, n
    # the viewer: welded to a mocap mount (no free joint), no actuators, no contacts
    mount = m.body(P + "viewer_mount").id
    vb = np.flatnonzero(m.body_rootid == mount)
    assert m.body_mocapid[mount] >= 0
    vj = [j for j in range(m.njnt) if m.jnt_bodyid[j] in vb]
    assert len(vj) == 45  # 6 legs x 7 + 3 neck
    assert all(m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE for j in vj)
    assert not any(m.actuator_trnid[a, 0] in vj for a in range(m.nu))
    vg = np.isin(m.geom_bodyid, vb)
    assert not np.any(m.geom_contype[vg]) and not np.any(m.geom_conaffinity[vg])
    # breakables are free bodies that don't touch the fly
    for nm, *_ in BREAKABLES:
        b = m.body(P + nm).id
        g = np.flatnonzero(m.geom_bodyid == b)
        assert m.jnt_type[m.body_jntadr[b]] == mj.mjtJoint.mjJNT_FREE
        assert not np.any(m.geom_conaffinity[g] & 8)  # FLY_BIT


def test_sequence_order_and_ponder_timing(run):
    assert run["seen"][:len(ORDER)] == ORDER, run["seen"]
    t = {}
    for p, tt in run["log"]:  # phase start times (first cycle)
        t.setdefault(p, tt)
    ponder = t["flick"] - t["ponder"]
    assert 0.75 - 0.002 <= ponder <= 1.0 + 0.002, ponder
    flick = t["flight"] - t["flick"]
    assert abs(flick - (run["job"].cfg.windup_s + run["job"].cfg.throw_s)) < 0.01
    # throw -> boom: the blast goes off almost at once (contact or the flight_s timer)
    assert t["flight"] < t["boom"] <= t["flight"] + run["job"].cfg.flight_s + 0.002
    job = run["job"]
    assert abs((t["rebuild"] - t["aftermath"]) - job.cfg.aftermath_s) < 0.01


def test_launch_goes_backward_over_the_shoulder(run):
    job, launch = run["job"], run["launch"]
    assert launch is not None
    lf = job.last_flight
    v, p0 = lf["v0"], lf["p0"]  # at the release
    assert v[0] < -50.0  # backward (the viewer faces +x)
    assert v[2] > 100.0  # up and over
    assert launch["v"][0] < 0 and launch["p"][0] < p0[0]  # the free plate really flies back
    assert lf["p_land"][0] < -3.0  # landed behind the chair, among the props
    # over the right shoulder: toward -y, the picture's left on the webcam (like the meme)
    assert v[1] < 0 and lf["p_land"][1] < p0[1]
    # a fast, flat throw: translation dominates, it spins about its own axis
    assert np.linalg.norm(v) > 150.0
    assert lf["flight_s"] <= job.cfg.flight_s + 0.002 and lf["by"] in ("contact", "timer")


def test_throw_snap_no_face_on_flip_body_twist_and_recoil(run):
    fx = run["fx"]
    # the plate never turns its face to the webcam (+x) during the flick
    assert fx["normal_x"] < 0.35, fx["normal_x"]
    # the torso twists (mocap mount) and the chair recoils (kinematic nudge) ...
    assert fx["twist"] > 15.0 and fx["chair"] > 0.05
    # ... and both are back at rest after the cycle
    job, d = run["job"], run["session"].sim.data
    assert np.allclose(d.mocap_quat[job.mount_mocap], (1, 0, 0, 0), atol=1e-6)
    assert np.linalg.norm(d.mocap_pos[job.chair_mocap]) < 1e-9


def test_edit_hit_stop_slowmo_and_flash(run):
    job, fx = run["job"], run["fx"]
    c = job.cfg
    # presentation clock: near-freeze first, then slow motion, then normal speed
    sc = fx["scales"]
    assert sc[0] == 1.0 and sc[1] == pytest.approx(c.hit_stop_scale)  # (the blast starts mid-chunk)
    assert c.slowmo_scale in [pytest.approx(v) for v in sc]
    assert fx["lag_at_boom"] == 0.0
    expect = (c.hit_stop_s * (1 - c.hit_stop_scale) + c.slowmo_s * (1 - c.slowmo_scale)
              + 0.5 * c.slowmo_ramp_s * (1 - c.slowmo_scale))
    assert fx["lag_end"] == pytest.approx(expect, abs=0.03)  # one blast's worth, no drift
    assert job._edit is None  # the edit ended with the cycle
    # post-process: identity (the same array) when idle, the impact flash at the blast
    assert fx["idle_same"]
    idle, impact = fx["frames"]["idle"], fx["frames"]["impact"]
    assert impact.shape == idle.shape
    assert impact.mean() > idle.mean() + 60  # a (near) white impact frame
    assert "EDIT FX" not in "\n".join(job.hud_lines())


def test_blast_moves_props_and_rebuild_resets_them(run):
    job, d = run["job"], run["session"].sim.data
    assert run["max_disp"] > 1.0  # the shockwave kick sent props flying
    assert job.n_props_launched >= 5
    assert job.n_rebuilds == 1
    cur = np.array([d.qpos[q:q + 3] for q in job.brk_q])
    assert np.max(np.linalg.norm(cur - run["home"], axis=1)) < 0.1  # back on their spots
    # the blast visuals are off again
    m = run["session"].sim.model
    assert m.mat_rgba[job.m_fire, 3] == 0.0 and d.mocap_pos[job.blast_mocap][2] < -10
    assert not np.any(m.geom_rgba[job.g_fire, 3]) and not np.any(m.geom_rgba[job.g_smoke, 3])
    assert m.light_castshadow[job.key_light] == job.key_shadow0


def test_counters_and_hud(run):
    job = run["job"]
    st = job.job_stats()
    assert st["plates_yeeted"] == 1 and job.work == 1.0
    assert st["explosions"] == 1
    assert st["viewers"] > job.cfg.viewers0
    assert st["vegetables_eaten"] == 0
    hud = "\n".join(job.hud_lines())
    assert "BROCCOLI TOSS FLY" in hud and "absolutely not" in hud
    assert "vegetables eaten: 0" in hud and "stream viewers" in hud
    assert job.n_falls == 0 and job.n_auto_recoveries == 0


def test_bitter_pulses_during_ponder(run):
    sent = [ev for ev, _ in run["brain"].sent]
    assert sent, "no bitter pulse sent during the ponder"
    assert all(ev.kind == "taste" and ev.details["tastes"] == ["bitter"] for ev in sent)
    assert "stand-in" in sent[0].details["label"]
    assert run["job"].n_bitter == len(sent)


def test_reset_mid_cycle(run):
    session, job, runner = run["session"], run["job"], run["runner"]
    while job.phase != "handover" and job.run_time() < 60:
        runner.step_chunk()
    assert job.phase == "handover"
    session.reset("manual")
    assert job.phase == "deliver" and job.n_resets_mid == 1
    d = session.sim.data
    cur = np.array([d.qpos[q:q + 3] for q in job.brk_q])
    assert np.max(np.linalg.norm(cur - run["home"], axis=1)) < 0.05
    # the plate rides on the host again, the host is back at the kitchen
    th = d.xpos[session.sim.thorax_body_id]
    assert np.linalg.norm(job.plate_pos()[:2] - th[:2]) < 0.6
    assert math.dist(job.fly_xy(), job.cfg.kitchen_xy) < 0.5


def test_presentation_clock_counts_held_steps():
    from fly_simulator.jobs import PresentationClock

    class J:
        def __init__(self):
            self.scale = 1.0

        def time_scale(self, dt):
            return self.scale

    j = J()
    clk = PresentationClock(j, 1e-4)
    assert clk.steps(150) == 150 and clk.lag == 0.0
    j.scale = 0.3
    n = sum(clk.steps(150) for _ in range(10))
    assert n == 450  # exactly 30 %, fractions carried over
    assert clk.lag == pytest.approx(0.105)
    j.scale = 0.0
    assert clk.steps(150) == 0 and clk.last_held_s == pytest.approx(0.015)
    j.scale = float("nan")  # a broken job never stalls the sim
    assert clk.steps(150) == 150
