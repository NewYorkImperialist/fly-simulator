"""Shared machinery for the scripted scenes that followed ``temple_standoff``
(``moon_landing``, ``kitty_hawk``, ``river_crossing``).

It reuses the temple scene's fly builder (``_make_fly``: a posed, kinematic
NeuroMechFly copy, tinted, scaled and dressed) and its ``FlyRig`` / ``Shot`` types,
and adds a small generic scene base class:

* ``PosedScene``: builds an MjSpec (settings, the subclass's set and cast), compiles
  it, calibrates every fly (stance height, joint signs, the recorded step's stride),
  and gives the subclass posing helpers (``set_fly``, ``legs_walking``, ``ik``);
  ``extra_geoms(t)`` lets a scene add per-frame particles (dust, snow, sand, mist) as
  MuJoCo scene geoms without touching the model;
* ``Post``: a screen-space post-process (bloom, the scene's own grade, vignette,
  grain, the 2.39:1 letterbox);
* ``SceneRenderer`` / ``render_scene``: the offscreen render loop piped to ffmpeg,
  with the scene's synthesized WAV muxed in.

Everything is a pure function of scene time: nothing is simulated.
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

from fly_simulator.jobs.geometry import quat_axis_angle, quat_mul
from fly_simulator.scenes.temple_standoff import (  # noqa: F401  (re-exported)
    AX3, DOFS, LEGS, TRIPOD_A, FlyLook, FlyRig, Shot, _make_fly, _quat_from_z, _quat_yaw, _smooth, _smoother,
    write_wav,
)

VIS = dict(contype=0, conaffinity=0, group=0, mass=0.0)


def shot_at(shots, t: float) -> Shot:
    for s in shots:
        if s.t0 <= t < s.t1:
            return s
    return shots[-1]


def quat_ypr(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """World quaternion of a fly (x forward): yaw about z, then nose-up ``pitch``,
    then ``roll`` (right side down for positive roll)."""
    q = _quat_yaw(yaw)
    if pitch:
        q = quat_mul(q, quat_axis_angle((0, 1, 0), -pitch))
    if roll:
        q = quat_mul(q, quat_axis_angle((1, 0, 0), roll))
    return np.array(q, float)


def quat_rotate(q, v) -> np.ndarray:
    R = np.empty(9)
    mj.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3) @ np.asarray(v, float)


def quat_to_mat(q) -> np.ndarray:
    R = np.empty(9)
    mj.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def ease_io(x: float) -> float:
    return _smoother(x)


def lerp(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


class PosedScene:
    """Base class. Subclasses set the class attributes and implement ``build(spec)``
    (the set, the props and the cast via ``add_fly``), ``setup()`` (after compile:
    IK, ids) and ``pose(t)``."""

    NAME = ""
    SHOTS: tuple[Shot, ...] = ()
    T_END = 0.0
    EXTENT = 40.0
    CENTER = (0.0, 0.0, 0.0)
    ZNEAR = 0.0025
    ZFAR = 3.0
    FOG = None  # (start, end) as fractions of the extent, with FOG_RGBA
    FOG_RGBA = (0.5, 0.5, 0.5, 1.0)
    HAZE = (0.02, 0.025, 0.035, 1.0)
    HEADLIGHT = ((0.02, 0.02, 0.02), (0.03, 0.03, 0.03))  # ambient, diffuse
    SKYBOX = False
    REFLECTION = False
    SHADOWCLIP = 0.6

    def __init__(self, seed: int = 0, shadows: bool = True):
        from flygym_demo.complex_terrain import PreprogrammedSteps

        self.seed = seed
        self.shadows = shadows
        self.steps = PreprogrammedSteps()
        self._fly_specs: dict[str, FlyLook] = {}
        spec = mj.MjSpec()
        spec.modelname = self.NAME
        self._settings(spec, shadows)
        self.build(spec)
        self.spec = spec
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        m = self.model
        self.rigs = {n: FlyRig(m, n, look) for n, look in self._fly_specs.items()}
        self._scratch = mj.MjData(m)
        self._calibrate()
        self.setup()
        self.pose(0.0)

    # ------------------------------------------------------------ build
    def _settings(self, spec, shadows: bool) -> None:
        spec.compiler.fusestatic = False
        spec.option.gravity = (0.0, 0.0, 0.0)
        spec.option.timestep = 1e-3
        spec.stat.extent = self.EXTENT
        spec.stat.center = self.CENTER
        v = spec.visual
        v.map.znear = self.ZNEAR
        v.map.zfar = self.ZFAR
        v.map.shadowclip = self.SHADOWCLIP
        v.map.shadowscale = 0.5
        if self.FOG is not None:
            v.map.fogstart, v.map.fogend = self.FOG
            v.rgba.fog = self.FOG_RGBA
        v.quality.shadowsize = 4096 if shadows else 0
        v.quality.offsamples = 8
        v.global_.offwidth = 1920
        v.global_.offheight = 1088
        v.global_.fovy = 40.0
        v.headlight.ambient = self.HEADLIGHT[0]
        v.headlight.diffuse = self.HEADLIGHT[1]
        v.headlight.specular = (0.0, 0.0, 0.0)
        v.rgba.haze = self.HAZE

    def build(self, spec) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def setup(self) -> None:
        pass

    def add_fly(self, spec, name: str, look: FlyLook, garments: dict | None = None,
                garment_tex: dict | None = None) -> None:
        root = _make_fly(name, look, garments or {}, garment_tex or {})
        mount = spec.worldbody.add_body(name=name + "_mount", mocap=True, pos=(0.0, 0.0, -50.0))
        site = mount.add_site(name=name + "_site")
        spec.attach(root, prefix=name + "/", site=site)
        self._fly_specs[name] = look

    # ------------------------------------------------------------ calibrate / IK
    def stand_angles(self) -> np.ndarray:
        return np.array([self.steps.get_joint_angles(leg, math.pi) for leg in LEGS])

    @staticmethod
    def geom_verts(m, d, g) -> np.ndarray:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    def _calibrate(self) -> None:
        m, d = self.model, self._scratch
        stand = self.stand_angles()
        for rig in self.rigs.values():
            rig.stand = stand.copy()
            d.qpos[:] = m.qpos0
            d.mocap_pos[rig.mocap] = (0, 0, 0)
            d.mocap_quat[rig.mocap] = (1, 0, 0, 0)
            d.qpos[rig.leg_q.ravel()] = stand.ravel()
            mj.mj_kinematics(m, d)
            rig.z_off = -min(float(self.geom_verts(m, d, g)[:, 2].min()) for g in rig.feet_geoms)
        if not self.rigs:
            return
        name, rig = next(iter(self.rigs.items()))
        d.qpos[:] = m.qpos0
        d.qpos[rig.leg_q.ravel()] = stand.ravel()
        mj.mj_kinematics(m, d)
        eye = m.body(f"{name}/l_eye").id
        e0 = d.xpos[eye].copy()
        d.qpos[rig.head_q[1]] = 0.3
        mj.mj_kinematics(m, d)
        self.head_up = -1.0 if d.xpos[eye][2] < e0[2] else 1.0
        d.qpos[rig.head_q[1]] = 0.0
        d.qpos[rig.head_q[0]] = 0.3
        mj.mj_kinematics(m, d)
        hc = d.xpos[rig.head]
        self.head_left = 1.0 if (d.xpos[eye] - hc)[0] < (e0 - hc)[0] else -1.0
        d.qpos[rig.head_q[0]] = 0.0
        self.wing_sign = {}
        for si, s in enumerate("lr"):
            wid = m.body(f"{name}/{s}_wing").id
            gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == wid][0]

            def tip():
                mj.mj_kinematics(m, d)
                return self.geom_verts(m, d, gid).mean(0)

            t0 = tip()
            res = {}
            for a in range(3):
                d.qpos[rig.wing_q[si, a]] = 0.4
                res[a] = tip() - t0
                d.qpos[rig.wing_q[si, a]] = 0.0
            ysign = 1.0 if s == "l" else -1.0
            sp = max(range(3), key=lambda a: abs(res[a][1]))
            li = max((a for a in range(3) if a != sp), key=lambda a: abs(res[a][2]))
            self.wing_sign[s] = dict(spread=(sp, 1.0 if res[sp][1] * ysign > 0 else -1.0),
                                     lift=(li, 1.0 if res[li][2] > 0 else -1.0))
        ph = np.linspace(0, 2 * math.pi, 181)
        strides = []
        for li_, leg in enumerate(LEGS):
            xs = []
            for p in ph:
                d.qpos[rig.leg_q[li_]] = self.steps.get_joint_angles(leg, p)
                mj.mj_kinematics(m, d)
                xs.append(d.xpos[rig.tarsus5[leg]][0])
            d.qpos[rig.leg_q[li_]] = stand[li_]
            xs = np.array(xs)
            i0 = int(np.argmin(np.abs(ph - self.steps.swing_period[leg][1])))
            strides.append(xs[i0] - xs[-1])
        self.stride_unit = float(np.mean(strides)) / rig.look.scale

    def ik(self, rig: FlyRig, leg: str, target_fly, q0: np.ndarray, iters: int = 300,
           legs: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Damped least squares on one leg (7 DoFs) with the mount at the origin:
        ``target_fly`` is the tarsus-5 point in the mount frame."""
        m, d = self.model, self._scratch
        li = LEGS.index(leg)
        d.qpos[:] = m.qpos0
        d.mocap_pos[rig.mocap] = (0, 0, 0)
        d.mocap_quat[rig.mocap] = (1, 0, 0, 0)
        d.qpos[rig.leg_q.ravel()] = (rig.stand if legs is None else legs).ravel()
        q = np.array(q0, float).copy()
        dadr = np.array([m.jnt_dofadr[np.flatnonzero(m.jnt_qposadr == a)[0]] for a in rig.leg_q[li]])
        jac = np.zeros((3, m.nv))
        bi = rig.tarsus5[leg]
        tgt = np.asarray(target_fly, float)
        err = math.inf
        for _ in range(iters):
            d.qpos[rig.leg_q[li]] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            e = tgt - d.xpos[bi]
            err = float(np.linalg.norm(e))
            if err < 2e-4:
                break
            mj.mj_jacBody(m, d, jac, None, bi)
            J = jac[:, dadr]
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3))
            dq = 0.5 * Jp @ e + 0.03 * (np.eye(7) - Jp @ J) @ (q0 - q)
            n = float(np.linalg.norm(dq))
            if n > 0.12:
                dq *= 0.12 / n
            q += dq
        return q, err

    def foot_fly(self, rig: FlyRig, leg: str, legs: np.ndarray | None = None) -> np.ndarray:
        """Tarsus-5 position of ``leg`` in the mount frame for a leg pose."""
        m, d = self.model, self._scratch
        d.qpos[:] = m.qpos0
        d.mocap_pos[rig.mocap] = (0, 0, 0)
        d.mocap_quat[rig.mocap] = (1, 0, 0, 0)
        d.qpos[rig.leg_q.ravel()] = (rig.stand if legs is None else legs).ravel()
        mj.mj_kinematics(m, d)
        return d.xpos[rig.tarsus5[leg]].copy()

    # ------------------------------------------------------------ posing
    def legs_walking(self, rig: FlyRig, phase: float, mag: float, skip=()) -> np.ndarray:
        q = rig.stand.copy()
        if mag <= 0:
            return q
        for li, leg in enumerate(LEGS):
            if leg in skip:
                continue
            p = phase + (0.0 if leg in TRIPOD_A else math.pi)
            q[li] = self.steps.get_joint_angles(leg, p, mag)
        return q

    def set_fly(self, rig: FlyRig, pos, quat, legs: np.ndarray, head=(0.0, 0.0, 0.0), wings=None,
                ant=None, wing_raw=None) -> None:
        """Mount at world ``pos`` with orientation ``quat``. ``head`` = (yaw left, up,
        roll); ``wings`` = (spread, lift) in calibrated directions, or ``wing_raw``
        (2, 3) joint angles."""
        d = self.data
        d.mocap_pos[rig.mocap] = pos
        d.mocap_quat[rig.mocap] = quat
        d.qpos[rig.leg_q.ravel()] = legs.ravel()
        yaw_h, up, roll = head
        d.qpos[rig.head_q] = (self.head_left * yaw_h, self.head_up * up, roll)
        wq = np.zeros((2, 3)) if wing_raw is None else np.array(wing_raw, float)
        if wings is not None:
            for si, s in enumerate("lr"):
                (a1, s1), (a2, s2) = self.wing_sign[s]["spread"], self.wing_sign[s]["lift"]
                wq[si, a1] += s1 * wings[0]
                wq[si, a2] += s2 * wings[1]
        d.qpos[rig.wing_q.ravel()] = wq.ravel()
        d.qpos[rig.ant_q.ravel()] = (np.zeros((2, 3)) if ant is None else ant).ravel()

    def set_fly_ground(self, rig: FlyRig, x: float, y: float, yaw: float, legs, ground: float = 0.0,
                       pitch: float = 0.0, roll: float = 0.0, dz: float = 0.0, **kw) -> None:
        self.set_fly(rig, (x, y, ground + rig.z_off + dz), quat_ypr(yaw, pitch, roll), legs, **kw)

    def park(self, rig: FlyRig) -> None:
        """Out of sight (below the set)."""
        self.set_fly(rig, (0.0, 0.0, -500.0), (1, 0, 0, 0), rig.stand)

    def pose(self, t: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def extra_geoms(self, t: float) -> list:
        """Per-frame scene geoms: (type, size3, pos3, mat9, rgba4, emission)."""
        return []

    def before_render(self, renderer: "SceneRenderer", t: float) -> None:
        """Hook: texture re-uploads and the like."""

    def shot(self, t: float) -> Shot:
        return shot_at(self.SHOTS, t)

    def camera(self, t: float) -> tuple[mj.MjvCamera, Shot]:
        sh = self.shot(t)
        eye, tgt = self.camera_pose(t, sh)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        v = tgt - eye
        cam.lookat[:] = tgt
        cam.distance = float(np.linalg.norm(v))
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        return cam, sh

    def camera_pose(self, t: float, sh: Shot) -> tuple[np.ndarray, np.ndarray]:
        """Eye / target for the shot (fixed; a subclass may add a tiny handheld sway)."""
        return np.array(sh.eye, float), np.array(sh.target, float)

    # the post-process: the scene's grade on float RGB in [0, 1+] (linear-ish)
    BLOOM = (0.72, 0.6)  # threshold (0..1), strength
    VIGNETTE = 0.32
    GRAIN_MONO = False

    def grade(self, x: np.ndarray, t: float, k: int, post: "Post") -> np.ndarray:
        return x

    def black(self, t: float) -> bool:
        return False


# ---------------------------------------------------------------------------
# post-process
# ---------------------------------------------------------------------------


class Post:
    def __init__(self, scene: PosedScene, width: int, height: int, letterbox: bool = True, grain: float = 3.0):
        self.scene = scene
        self.W, self.H = width, height
        self.letterbox = letterbox
        self.grain = grain
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self.r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        self.vig = (1.0 - scene.VIGNETTE * np.clip(self.r2 - 0.15, 0, None) ** 1.2)[..., None].astype(np.float32)
        self.bar = int(round((height - width / 2.39) / 2)) if letterbox else 0

    def blur(self, x: np.ndarray, sigma_1080: float) -> np.ndarray:
        """Gaussian blur, the sigma given in 1080p pixels (done at 1/4 size)."""
        import cv2

        W, H = self.W, self.H
        s4 = cv2.resize(x, (max(W // 4, 1), max(H // 4, 1)), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(s4, (0, 0), max(sigma_1080 * H / 1080 / 4, 0.3))
        return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)

    def __call__(self, frame: np.ndarray, t: float, k: int) -> np.ndarray:
        import cv2

        sc = self.scene
        if sc.black(t):
            return np.zeros_like(frame)
        W, H = self.W, self.H
        x = frame.astype(np.float32) / 255.0
        thr, strength = sc.BLOOM
        if strength > 0:
            bright = np.maximum(x - thr, 0.0)
            glow = self.blur(bright, 12.0) + 1.6 * self.blur(bright, 30.0)
            x = x + strength * glow
        x = sc.grade(x, t, k, self)
        x = x * self.vig
        if self.grain > 0:
            rng = np.random.default_rng(1000 + k)
            if sc.GRAIN_MONO:
                n = rng.normal(0.0, self.grain / 255.0, (H // 2, W // 2)).astype(np.float32)
                x = x + cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            else:
                n = rng.normal(0.0, self.grain / 255.0, (H // 2, W // 2, 3)).astype(np.float32)
                n = 0.7 * n.mean(2, keepdims=True) + 0.3 * n
                x = x + cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
        out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        if self.bar > 0:
            out[:self.bar] = 0
            out[H - self.bar:] = 0
        return out


class SceneRenderer:
    def __init__(self, scene: PosedScene, width: int, height: int, letterbox: bool = True, grain: float = 3.0):
        self.scene = scene
        self.W, self.H = width, height
        self.r = mj.Renderer(scene.model, height, width, max_geom=30000)
        self.r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = int(scene.REFLECTION)
        self.r.scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = int(scene.SKYBOX)
        self.r.scene.flags[mj.mjtRndFlag.mjRND_FOG] = int(scene.FOG is not None)
        self.post = Post(scene, width, height, letterbox, grain)
        self.opt = mj.MjvOption()

    @property
    def context(self):
        return self.r._mjr_context

    def frame(self, t: float, k: int) -> np.ndarray:
        sc = self.scene
        if sc.black(t):
            return np.zeros((self.H, self.W, 3), np.uint8)
        sc.pose(t)
        cam, sh = sc.camera(t)
        sc.model.vis.global_.fovy = sh.fovy
        sc.before_render(self, t)
        self.r.update_scene(sc.data, camera=cam, scene_option=self.opt)
        scn = self.r.scene
        for typ, size, pos, mat, rgba, emi in sc.extra_geoms(t):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(g, int(typ), np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
            g.emission = emi
            g.category = int(mj.mjtCatBit.mjCAT_DECOR)
            scn.ngeom += 1
        raw = self.r.render()
        return self.post(raw, t, k)

    def close(self) -> None:
        self.r.close()


@dataclass
class RenderOptions:
    width: int = 1920
    height: int = 1080
    fps: float = 24.0
    letterbox: bool = True
    audio: bool = True
    shadows: bool = True
    grain: float = 3.0
    t0: float = 0.0
    t1: float = 0.0  # 0 = the whole scene
    png_dir: str | None = None
    png_every: int = 0
    crf: int = 18
    caption: str | None = None  # unused (no subtitles in these scenes)
    caption_t: tuple[float, float] = (0.0, 0.0)


def render_scene(scene_cls, synth_audio, out: str | Path, opt: RenderOptions, log=print) -> dict:
    import cv2

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    scene = scene_cls(shadows=opt.shadows)
    name = scene_cls.NAME
    t1 = opt.t1 if opt.t1 > 0 else scene_cls.T_END
    log(f"[{name}] built in {time.perf_counter() - t_start:.1f} s (nbody {scene.model.nbody}, "
        f"ngeom {scene.model.ngeom})")
    W, H = opt.width, opt.height
    srend = SceneRenderer(scene, W, H, opt.letterbox, opt.grain)
    wav = None
    if opt.audio:
        wav = out.with_suffix(".wav")
        a = synth_audio(scene)
        i0, i1 = int(opt.t0 * 48000), int(t1 * 48000)
        write_wav(wav, a[i0:i1])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(opt.fps), "-i", "-"]
    if wav is not None:
        cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(opt.crf), "-preset", "medium",
            "-movflags", "+faststart", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n0, n1 = int(round(opt.t0 * opt.fps)), int(round(t1 * opt.fps))
    png = Path(opt.png_dir) if opt.png_dir else None
    if png:
        png.mkdir(parents=True, exist_ok=True)
    t_r = time.perf_counter()
    try:
        for k in range(n0, n1):
            t = k / opt.fps
            frame = srend.frame(t, k)
            ff.stdin.write(frame.tobytes())
            if png and opt.png_every and (k - n0) % opt.png_every == 0:
                cv2.imwrite(str(png / f"f{k:04d}_{t:05.2f}.png"), frame[..., ::-1])
    finally:
        ff.stdin.close()
        ff.wait()
        srend.close()
        if wav is not None and wav.exists():
            wav.unlink()
    dt = time.perf_counter() - t_r
    log(f"[{name}] {n1 - n0} frames {W}x{H} @ {opt.fps:g} fps in {dt:.1f} s -> {out}")
    return dict(frames=n1 - n0, seconds=dt, out=str(out))


# ---------------------------------------------------------------------------
# audio helpers (numpy / scipy)
# ---------------------------------------------------------------------------


class Audio:
    """A mono float64 buffer with small synthesis helpers."""

    def __init__(self, dur: float, sr: int = 48000, seed: int = 0):
        self.sr = sr
        self.n = int(dur * sr)
        self.t = np.arange(self.n) / sr
        self.out = np.zeros(self.n)
        self.rng = np.random.default_rng(seed)

    def lp(self, x, fc, order=2):
        import scipy.signal as sg

        b, a = sg.butter(order, min(fc / (self.sr / 2), 0.99), "low")
        return sg.lfilter(b, a, x)

    def hp(self, x, fc, order=2):
        import scipy.signal as sg

        b, a = sg.butter(order, fc / (self.sr / 2), "high")
        return sg.lfilter(b, a, x)

    def bp(self, x, f0, f1, order=2):
        import scipy.signal as sg

        b, a = sg.butter(order, [f0 / (self.sr / 2), min(f1 / (self.sr / 2), 0.99)], "band")
        return sg.lfilter(b, a, x)

    @staticmethod
    def db(v):
        return 10 ** (v / 20)

    @staticmethod
    def norm(x):
        return x / (np.abs(x).max() + 1e-9)

    def noise(self, k):
        return self.rng.normal(0, 1, k)

    def add(self, ev, at: float, gain: float = 1.0):
        i = int(at * self.sr)
        if i >= self.n or i + len(ev) <= 0:
            return
        j = min(self.n, i + len(ev))
        s = max(0, -i)
        self.out[max(i, 0):j] += gain * ev[s:j - i]

    def tick(self, level_db, dur=0.03, f0=900.0, f1=4200.0, decay=0.006):
        k = int(dur * self.sr)
        e = self.bp(self.noise(k), f0, f1) * np.exp(-np.arange(k) / (decay * self.sr))
        return self.db(level_db) * self.norm(e)

    def fade_out(self, at: float, dur: float = 0.006):
        i = int(at * self.sr)
        f = int(dur * self.sr)
        i0 = max(0, i - f)
        self.out[i0:i] *= np.linspace(1, 0, i - i0)
        self.out[i:] = 0.0

    def result(self) -> np.ndarray:
        return np.clip(self.out, -1.0, 1.0).astype(np.float32)
