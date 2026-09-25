"""App <-> brain integration (perpetualfly/brain_link.py), with the synthetic brain."""

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from perpetualfly import AppConfig
from perpetualfly.app import Session, build_arg_parser, config_from_args, main, parse_script_keys
from perpetualfly.brain_link import (FETCH_COMMAND, BrainLink, BrainLinkConfig, combine_drive,
                                     hit_side, missing_requirements)

pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[1]
SYN = {"n": 50, "p_conn": 0.1, "seed": 0}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); ask ps
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _wait_dead(pids, timeout=10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(_pid_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


# ----------------------------------------------------------------------------- CLI
def test_cli_brain_flags():
    parse = lambda *a: config_from_args(build_arg_parser().parse_args(list(a))).brain  # noqa: E731
    b = parse()
    assert not b.enabled and not b.steer
    b = parse("--brain")
    assert b.enabled and b.window and not b.steer
    b = parse("--brain-headless")
    assert b.enabled and not b.window
    b = parse("--brain", "--no-brain-window")
    assert b.enabled and not b.window
    b = parse("--brain-steer")  # implies --brain
    assert b.enabled and b.steer and b.window
    assert parse_script_keys("5:o, 2:left,2.5:SPACE") == [(2.0, "left"), (2.5, "space"), (5.0, "o")]


def test_missing_data_message(tmp_path, capsys):
    msg = missing_requirements(BrainLinkConfig(enabled=True, data_dir=str(tmp_path)))
    assert msg is not None and FETCH_COMMAND in msg and str(tmp_path) in msg
    assert missing_requirements(BrainLinkConfig(synthetic=SYN)) is None
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"brain": {"data_dir": str(tmp_path / "nope")}}))
    assert main(["--config", str(cfg), "--brain", "--headless", "--no-log"]) == 2
    assert FETCH_COMMAND in capsys.readouterr().err


# ------------------------------------------------------------------- pure helpers
def test_combine_drive_math():
    hold = np.array([1.2, 0.8])
    # quiet brain: exactly the heading hold
    assert np.array_equal(combine_drive(hold, np.ones(2)), hold)
    # forward-walking brain: hold + (brain - 1)
    b = np.array([1.1, 1.3])
    assert np.allclose(combine_drive(hold, b), hold + b - 1.0)
    # stopping brain: heading correction scaled by the mean drive
    assert np.allclose(combine_drive(hold, np.array([0.5, 0.5])), [0.6, 0.4])
    # reversing brain: no heading correction, pure brain command
    assert np.allclose(combine_drive(hold, np.array([-1.0, -0.8])), [-1.0, -0.8])
    # clipped to the controller limits
    assert np.allclose(combine_drive(np.array([1.4, 0.6]), np.array([1.5, 1.5])), [1.5, 1.1])


def test_hit_side_in_fly_frame():
    h = math.radians(90)  # fly faces +y; its left is -x
    assert hit_side((1, 0, 0), h) == "left"  # pushed to its right -> hit on the left
    assert hit_side((-1, 0, 0), h) == "right"
    assert hit_side((0, -1, 0), h) == "front"  # pushed backward
    assert hit_side((0, 1, 0), h) == "rear"
    assert hit_side((0.1, 0, -1), h) == "top"
    assert hit_side((0, 0, 0), h) == "none"


# ------------------------------------------------------------------- session
def _session(tmp_path, steer: bool, log: bool = True):
    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path)
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=steer, synthetic=SYN)
    link = BrainLink(cfg.brain, headless=True, say=lambda m: None)
    link.wait_ready(60)
    s = Session(cfg, log=log, say=lambda m: None, brain=link)
    return s, link


def _run(s, n_chunks, steps=150):
    for _ in range(n_chunks):
        s.sim.step(steps)
        s.after_physics()


def test_events_forwarded_logged_and_brain_follows_fly_time(tmp_path):
    s, link = _session(tmp_path, steer=False)
    try:
        assert s.sim.controller.signal_filter is None  # not steering
        _run(s, 20)
        # whip hit from the fly's left (the whip is the default hit mode)
        s.handle_whip_key("left")
        for _ in range(60):
            _run(s, 1)
            if any(e.kind == "whip_hit" for e in link.stim_log):
                break
        whip = [e for e in link.stim_log if e.kind == "whip_hit"]
        assert whip and whip[0].side == "left" and 0 < whip[0].intensity <= 1
        assert "body" in whip[0].details
        # shove to the fly's right -> hit on its left side
        s.handle_whip_key("h")
        s.handle_whip_key("right")
        shove = [e for e in link.stim_log if e.kind == "shove"]
        assert shove and shove[-1].side == "left"
        # manual stimuli (keys O / T)
        assert link.handle_key("o").startswith("[brain] loom")
        assert link.handle_key("t").startswith("[brain] sugar")
        assert {e.details.get("set") for e in link.stim_log if e.kind == "manual"} >= {"LC4", "sugar"}
        # reset -> brain reset (logged)
        s.reset("manual")
        # wait for states; the brain is paced to the fly's run time
        deadline = time.time() + 20
        while time.time() < deadline and (link.latest is None or link.latest.sim_time is None
                                          or link.latest.sim_time < s.run_time() - 0.25):
            _run(s, 1)
            time.sleep(0.01)
        st = link.latest
        assert st is not None and st.sim_time <= s.run_time() + 1e-6
        assert link.lag() is not None and link.lag() < 0.5
        assert link.status_line().startswith("BRAIN t")
        # pausing: no clock marks -> the brain stops at the fly's time
        t_fly = s.run_time()
        time.sleep(0.5)
        link.update()
        assert link.latest.sim_time <= t_fly + 1e-6
    finally:
        s.close("test")
        link.close()
    run_dir = s.logger.run_dir
    rows = list(csv.DictReader(open(run_dir / "events.csv")))
    kinds = [r["event_type"] for r in rows]
    assert kinds.count("brain_stim") == len(link.stim_log) and "brain_reset" in kinds
    m = list(csv.DictReader(open(run_dir / "metrics.csv")))
    assert {"brain_time", "brain_lag", "brain_drive_L", "dn_escape", "mn9_hz"} <= set(m[0])
    assert any(r["brain_time"] not in ("", "nan") for r in m)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["brain"]["process"]["pace"] == "sim" and config["app"]["brain"]["enabled"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["brain"]["brain_states"] > 0


def test_brain_drive_applied_only_with_steer(tmp_path):
    s, link = _session(tmp_path, steer=True, log=False)
    pids = link.pids
    try:
        ctl = s.sim.controller
        assert ctl.signal_filter is not None
        _run(s, 5)
        # quiet brain: identical to heading hold
        f, ctl.signal_filter = ctl.signal_filter, None
        hold = ctl.descending_signal()
        ctl.signal_filter = f
        assert np.allclose(ctl.descending_signal(), hold)
        # a brain drive (as if MDN fired): the controller gets combine_drive(...)
        link.drive[:] = [-0.8, -0.8]
        link._w = 0.0
        assert np.allclose(ctl.descending_signal(), combine_drive(hold, link.drive))
        link.drive[:] = [0.6, 0.6]
        link._w = 0.6
        assert np.allclose(ctl.descending_signal(), combine_drive(hold, link.drive))
    finally:
        s.close("test")
        link.close()
    assert s.sim.controller.signal_filter is None
    assert _wait_dead(pids)


def test_no_orphans_after_exit_and_parent_kill(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"brain": {"synthetic": SYN}, "terrain": {"difficulty": "flat"}}))
    base = [sys.executable, str(ROOT / "scripts" / "run_sim.py"), "--config", str(cfg),
            "--headless", "--brain-headless", "--no-log"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # 1) normal exit
    out = subprocess.run(base + ["--max-seconds", "0.3"], capture_output=True, text=True,
                         timeout=180, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("brain:"))
    pid = int(line.split("(pid ")[1].split(")")[0])
    assert _wait_dead([pid], 5)
    # 2) parent killed with SIGKILL (no cleanup at all)
    p = subprocess.Popen(base + ["--max-seconds", "1000"], stdout=subprocess.PIPE, text=True,
                         env=env)
    try:
        pid = None
        deadline = time.time() + 120
        while time.time() < deadline:
            ln = p.stdout.readline()
            if ln.startswith("brain:"):
                pid = int(ln.split("(pid ")[1].split(")")[0])
                break
        assert pid is not None and _pid_alive(pid)
        time.sleep(0.5)
        p.send_signal(signal.SIGKILL)
        p.wait(10)
        assert _wait_dead([pid], 10)
    finally:
        if p.poll() is None:
            p.kill()
