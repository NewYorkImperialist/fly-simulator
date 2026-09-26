"""The sensory screen (perpetualfly.brain.screen) runs on the synthetic network."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("numba")
pytest.importorskip("scipy")

from perpetualfly.brain.engine import LIFEngine  # noqa: E402
from perpetualfly.brain.process import _synthetic_table  # noqa: E402
from perpetualfly.brain.schema import DESCENDING_GROUPS  # noqa: E402
from perpetualfly.brain.screen import (KEY_DN_COLS, _jo_fine, behaviour_max,  # noqa: E402
                                       group_hops, min_hops_to, run_screen,
                                       sensory_groups)


def test_jo_fine_names():
    assert _jo_fine("JO-A1") == "JO-A"
    assert _jo_fine("JO-B1_a") == "JO-B"
    assert _jo_fine("JO-CM") == "JO-CM"
    assert _jo_fine("JO-EV3") == "JO-EV"
    assert _jo_fine("JO-mz") == "JO-other"
    assert _jo_fine("BM_Ant") is None


def test_min_hops_on_chain():
    from perpetualfly.brain.engine import csr_from_edges

    # 0 -> 1 -> 2 -> 3 (excitatory), 4 -| 3 (inhibitory: not a path), 5 -> 3 (1 synapse)
    pre = np.array([0, 1, 2, 4, 5])
    post = np.array([1, 2, 3, 3, 3])
    w = np.array([5, 5, 5, -9, 1])
    ip, ind, ww = csr_from_edges(pre, post, w, 6)
    d = min_hops_to(ip, ind, ww, np.array([3]), 6)
    assert d.tolist() == [3, 2, 1, 0, -1, 1]
    d5 = min_hops_to(ip, ind, ww, np.array([3]), 6, min_syn=5)
    assert d5[5] == -1 and d5[0] == 3


def test_screen_runs_on_synthetic_network():
    table, (ip, ind, w) = _synthetic_table(50)
    groups = sensory_groups(table)
    names = {g.name for g in groups}
    # synthetic sensory neurons 0..3 are SA_DMT_DMetaN, two per side
    assert {"SA_DMT_DMetaN:left", "SA_DMT_DMetaN:right", "body_mech:left"} <= names
    assert all(g.whip_plausible for g in groups)
    eng = LIFEngine(ip, ind, w, seed=0)
    rows = run_screen(eng, table, groups, rates=(200.0,), seconds=0.1, trials=1)
    assert len(rows) == len(groups)
    for r in rows:
        for k in (*DESCENDING_GROUPS, "MN9", *KEY_DN_COLS):
            assert r[k] >= 0.0
        assert set(behaviour_max(r)) == {"walk", "turn", "backward", "escape", "groom", "MN9"}
        assert isinstance(r["top_dn"], str)
        assert r["persist_sps"] >= 0.0 and isinstance(r["runaway"], bool)
    # driven input recruits something downstream in this dense random network
    assert max(r["n_active"] for r in rows) > 0
    hops = group_hops(ip, ind, w, table, groups)
    assert set(hops) == names
    assert all(set(h) >= {"walk", "turn", "escape"} for h in hops.values())
