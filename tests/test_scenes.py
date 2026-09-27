"""The scripted scenes (docs/SCENES.md): build, timeline, the blade light."""

from __future__ import annotations

import numpy as np
import pytest

from fly_simulator.scenes import SCENES, load_scene
from fly_simulator.scenes import temple_standoff as TS


@pytest.fixture(scope="module")
def scene():
    return TS.TempleStandoff(shadows=False)


def test_registry():
    assert "temple_standoff" in SCENES
    assert load_scene("temple_standoff") is TS


def test_builds(scene):
    m = scene.model
    assert set(scene.rigs) == {"tall"} | {lf.name for lf in TS.LITTLE}
    assert 5 <= len(TS.LITTLE) <= 7
    for name in ("tall/garment_cape", "tall/garment_hood", "lead/garment_tunic", "lead/garment_sash"):
        assert m.geom(name).id >= 0
    assert max(scene.ik_err.values()) < 1e-3
    # sizes: the tall fly towers over the little ones
    scene.pose(9.0)
    d = scene.data
    thorax_z = lambda n: d.xpos[scene.rigs[n].thorax][2]  # noqa: E731
    assert thorax_z("tall") - TS.DAIS_Z > 2.0 * thorax_z("lead")


def test_shot_timeline_ordered():
    shots = TS.SHOTS
    assert shots[0].t0 == 0.0 and shots[-1].t1 == TS.T_END
    for a, b in zip(shots, shots[1:]):
        assert a.t1 == b.t0 and a.t0 < a.t1
    assert TS.shot_at(TS.T_IGNITE).name == "handle_ignition"
    assert TS.shot_at(TS.T_BLACK).name == "black"
    assert (TS.T_WALK0 < TS.T_NOTICE < TS.T_STEP0 < TS.T_HANDLE0 < TS.T_IGNITE < TS.T_RECOIL
            < TS.T_LEAD_RECOIL < TS.T_BLACK < TS.T_END)


def test_blade_light_off_before_on_after(scene):
    m, li = scene.model, scene.blade_light
    core = scene.m_blade["blade_core"]
    for t in (0.0, 5.0, 10.9, TS.T_IGNITE - 1e-3):
        scene.pose(t)
        assert not scene.blade_light_on(t)
        assert np.all(m.light_diffuse[li] == 0.0)
        assert m.mat_rgba[core][3] == 0.0
    for t in (TS.T_IGNITE, 11.2, 12.0):
        scene.pose(t)
        assert scene.blade_light_on(t)
        assert m.light_diffuse[li].max() > 1.0  # the brightest light in the scene
        assert m.mat_rgba[core][3] == 1.0
    assert scene.blade_progress(TS.T_IGNITE + TS.IGNITE_S) == 1.0


def test_recoil_moves_back(scene):
    scene.pose(TS.T_RECOIL - 0.05)
    p0 = scene.data.mocap_pos[scene.rigs["l1"].mocap].copy()
    lead0 = scene.data.mocap_pos[scene.rigs["lead"].mocap].copy()
    scene.pose(TS.T_RECOIL + 0.25)
    p1 = scene.data.mocap_pos[scene.rigs["l1"].mocap].copy()
    lead1 = scene.data.mocap_pos[scene.rigs["lead"].mocap].copy()
    assert p1[0] < p0[0] - 0.3  # away from the tall fly (+x)
    assert np.allclose(lead0[:2], lead1[:2])  # the lead freezes a beat first


def test_audio_silent_after_cut():
    sr = 48000
    a = TS.synth_audio(None, sr=sr)
    assert len(a) == int(TS.T_END * sr)
    assert np.abs(a[int(TS.T_BLACK * sr):]).max() == 0.0
    ig = int(TS.T_IGNITE * sr)
    assert np.abs(a[ig:ig + 4800]).max() > 5 * np.abs(a[ig - 9600:ig - 4800]).max()
