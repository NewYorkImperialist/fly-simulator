"""Rebuild ``data/grooming_front_legs.npz`` from the NeuroMechFly v1 recording.

Source: NeLy-EPFL/NeuroMechFly (Lobato-Rios et al. 2022),
``data/joint_tracking/grooming/fly1/df3d/joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl``
-- a tethered fly whose antennal descending neurons (aDN) were activated
optogenetically (CsChrimson), recorded with DeepFly3D, joint angles at 2 kHz
(NeuroMechFly's replay used time_step 5e-4), 17972 samples. 0-3 s = front-leg
grooming (front legs move 7-12 deg std, mid/hind legs < 0.3 deg), 6-9 s = walking.

Conversion: the DeepFly3D angles use the legacy NeuroMechFly joint convention
(``flygym/assets/model/neuromechfly/legacy/flygym1_deepfly3d_rollyawpitch.xml``,
keys ThC_yaw/pitch/roll, CTr_pitch/roll, FTi_pitch, TiTa_pitch ->
joint_<LEG>Coxa_yaw, Coxa, Coxa_roll, Femur, Femur_roll, Tibia, Tarsus1, exactly as
NeuroMechFly's kinematic_replay.py maps them). The legacy model's leg body frames
coincide with FlyGym 2.1's at zero angles (checked: same positions, 0 deg
rotation difference). For every frame we compute the legacy forward kinematics and
solve (least squares) for the FlyGym 2.1 yaw-pitch-roll leg angles that reproduce
the coxa / femur / tibia / tarsus1 orientations and the tarsus positions relative
to the thorax. Residual: tarsus5 position error <= 0.7 um.

    .venv/bin/python -m fly_simulator.actions.convert_grooming --workdir /tmp/groom
"""

from __future__ import annotations

import argparse
import pickle
import re
import urllib.request
import warnings
from pathlib import Path

import mujoco as mj
import numpy as np

URL = ("https://raw.githubusercontent.com/NeLy-EPFL/NeuroMechFly/main/data/joint_tracking/"
       "grooming/fly1/df3d/joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl")
KEYS = {"ThC_pitch": "Coxa", "ThC_yaw": "Coxa_yaw", "ThC_roll": "Coxa_roll", "CTr_pitch": "Femur",
        "CTr_roll": "Femur_roll", "FTi_pitch": "Tibia", "TiTa_pitch": "Tarsus1"}
SEGS = ["coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus5"]
LSEG = {"coxa": "Coxa", "trochanterfemur": "Femur", "tibia": "Tibia", "tarsus1": "Tarsus1",
        "tarsus5": "Tarsus5"}


def legacy_model() -> mj.MjModel:
    """The legacy DeepFly3D-convention model, meshes replaced by spheres
    (kinematics only)."""
    import flygym

    path = Path(flygym.__file__).parent / "assets/model/neuromechfly/legacy/flygym1_deepfly3d_rollyawpitch.xml"
    s = path.read_text()
    s = re.sub(r"<mesh [^>]*/>", "", s)
    s = re.sub(r'mesh="[^"]*"', 'size="0.01"', s).replace('type="mesh"', 'type="sphere"')
    s = s.replace('fusestatic="true"', 'fusestatic="false"')
    return mj.MjModel.from_xml_string(s)


def convert(raw: dict, legs=("lf", "rf"), t0=0.0, t1=3.0, src_hz=2000, out_hz=200):
    from scipy.optimize import least_squares
    from flygym_demo.complex_terrain import make_locomotion_fly

    ml = legacy_model()
    dl = mj.MjData(ml)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fly = make_locomotion_fly()
        mv, dv = fly.compile()
    dof_order = fly.get_actuated_jointdofs_order("position")

    def bid(m, n):
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)

    def jadr(m, n):
        return m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]

    tl, tv = bid(ml, "Thorax"), bid(mv, "c_thorax")
    idx = np.arange(int(t0 * src_hz), int(t1 * src_hz), src_hz // out_hz)
    cols, names, max_err = [], [], 0.0
    for leg in legs:
        L = leg.upper()
        ladr = {k: jadr(ml, f"joint_{L}{v}") for k, v in KEYS.items()}
        lb = [bid(ml, f"{L}{LSEG[s]}") for s in SEGS]
        vb = [bid(mv, f"{leg}_{s}") for s in SEGS]
        vdofs = [d for d in dof_order if d.child.pos == leg]
        vadr = np.array([jadr(mv, d.name) for d in vdofs])

        def frames(m, d, root, bodies):
            R0 = d.xmat[root].reshape(3, 3)
            return [(R0.T @ d.xmat[b].reshape(3, 3), R0.T @ (d.xpos[b] - d.xpos[root])) for b in bodies]

        q = np.array([fly.jointdof_to_neutralangle[d] for d in vdofs])
        out = []
        for i in idx:
            dl.qpos[:] = 0
            for k, a in ladr.items():
                dl.qpos[a] = raw[f"{L}_leg"][k][i]
            mj.mj_kinematics(ml, dl)
            tgt = frames(ml, dl, tl, lb)

            def resid(q):
                dv.qpos[:] = 0
                dv.qpos[vadr] = q
                mj.mj_kinematics(mv, dv)
                cur = frames(mv, dv, tv, vb)
                r = [(c[0] - t[0]).ravel() for c, t in zip(cur[:4], tgt[:4])]
                r += [(c[1] - t[1]) * 2.0 for c, t in zip(cur, tgt)]
                return np.concatenate(r)

            q = least_squares(resid, q, xtol=1e-10, ftol=1e-10).x
            out.append(q.copy())
            r = resid(q)
            max_err = max(max_err, float(np.linalg.norm(r[-3:]) / 2.0))
        cols.append(np.array(out))
        names += [d.name for d in vdofs]
    return np.concatenate(cols, axis=1), names, max_err


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, required=True, help="where the 23 MB source pickle goes")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "data/grooming_front_legs.npz")
    args = p.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    src = args.workdir / "grooming_joint_angles_df3d.pkl"
    if not src.exists():
        urllib.request.urlretrieve(URL, src)
    with open(src, "rb") as f:
        raw = pickle.load(f)
    angles, names, err = convert(raw)
    np.savez_compressed(args.out, angles=angles.astype(np.float32), dof_names=np.array(names), fps=200,
                        source=f"{URL} t=0-3 s, 2 kHz -> 200 Hz, DeepFly3D (legacy NMF) -> FlyGym 2.1 "
                               "yaw-pitch-roll via per-frame IK on segment orientations/positions")
    print(f"wrote {args.out} {angles.shape}, max tarsus5 error {err * 1e3:.2f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
