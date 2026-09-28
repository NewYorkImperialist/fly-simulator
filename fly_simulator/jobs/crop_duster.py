"""CROP DUSTER: the fly crop-dusts fields forever (docs/JOBS.md, "crop_duster").

The second job on the **real flight fly** (``needs_flight = True``, like delivery_pilot:
flapping wings in MuJoCo's fluid model, dt 5e-5 s, docs/FLIGHT.md). Every bit of lift,
thrust and steering comes from the beating wings; no external force acts on the fly.

* **Scene** (fly scale): a patchwork of four fields north of a little grass airstrip
  (the fly spawns on it), each with crop rows along x, post-and-rail fences, a red barn,
  a silo, a windpump whose rotor turns with the wind, a hangar, the dust hopper (the
  refill station), a windsock, trees and hedgerows. All visual: the fly flies over it.
* **Loop** (``phase``): take-off with ``FlightMode``'s real one (the Jump -> wings ->
  hover) -> climb to the transit altitude and turn -> **transit** (velocity control
  along the bearing, trapezoid speed profile, like delivery_pilot's cruise) to the
  first row of the field whose crop has grown back the most -> **line-up** (the
  position loop over the row start, descend to the pass altitude, turn to the row
  heading) -> **pass** (velocity control along the row at ``pass_alt``, heading =
  pure pursuit onto the row line so the wind's cross-drift is corrected, the dust on)
  -> at the row end a **pull-up turn** (brake, climb ``pullup`` mm, hover-turn 180 deg
  while the position loop slides over to the next row) -> the next pass ... After the
  last row the next field, or, when the hopper can't do another field, back to the
  airstrip: approach, ``FlightMode``'s landing, **refill** (kinematic, labelled),
  take-off again.
* **Dust** (visual, labelled): while spraying, puffs from a fixed pool of mocap
  particles leave the fly, drift with the (physical) wind and settle; the crop cells
  under the settling plume (downwind of the flight track by wind x fall time) get a
  dust dose, and their material steps through 5 stages from green to pale dusted.
  The crop "grows back": the dose fades over ``regrow_s``, so no field stays done.
* **Wind is physical**: MuJoCo's medium velocity (``model.opt.wind``): a slow wandering
  breeze plus random **gusts** (counted), acting on the wings and body through the
  fluid model; the flight controller rejects it.

Counters: fields dusted (the work counter), rows, flight distance and time, refills,
wind gusts, dust used, landings, crash landings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import cv2
import mujoco as mj
import numpy as np

from fly_simulator.jobs import crop_duster_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, wrap_angle
from fly_simulator.jobs.registry import register_job

P = "dust/"


@dataclass
class CropDusterConfig(JobConfig):
    update_every_steps: int = 10  # 0.5 ms at the flight timestep
    stuck_timeout_s: float = 90.0  # no row / field / refill this long -> explicit reset
    # --- the farm (mm) -------------------------------------------------------------
    field_len: float = 36.0  # along x (the rows)
    n_rows: int = 6
    row_gap: float = 3.0
    seg_len: float = 1.5  # crop cells per row: field_len / seg_len
    fields: tuple = ((-20.0, 22.0), (20.0, 22.0), (20.0, 46.0), (-20.0, 46.0))  # field centres
    # --- flight guidance (our code; the flight itself is FlightMode + HoverController) ---
    pass_alt: float = 3.2  # COM height above the ground on a dusting pass
    pullup: float = 1.2  # climb this much while braking at a row end
    transit_alt: float = 10.0
    pass_speed: float = 55.0  # mm/s
    transit_speed: float = 90.0
    accel: float = 300.0  # mm/s^2, speed command ramp up
    decel: float = 220.0  # mm/s^2, braking at a row end / before a waypoint
    lookahead: float = 7.0  # pure pursuit onto the row line (mm)
    lead_in: float = 6.0  # a pass starts this far before the field edge
    overrun: float = 0.5  # ... and ends this far past the far edge (then it brakes and turns)
    xtrack_ki: float = 2.5  # 1/s: integral on the cross-track error of a pass (wind)
    turn_rate: float = 3.5  # rad/s, heading slew in the air (FlightMode.turn_rate)
    turn_first_deg: float = 20.0
    cruise_xy_zeta: float = 3.0  # velocity-loop damping while flying passes / transit
    handover_mm: float = 1.5
    descend_rate: float = 8.0  # mm/s, the fastest commanded descent in a hover
    approach_alt: float = 4.2  # COM height over the strip before the landing
    lineup_tol: float = 1.0  # mm, position error to start a pass / a row after a turn
    lineup_yaw_deg: float = 15.0
    lineup_timeout_s: float = 2.5
    approach_xy_wn: float = 7.0  # position loop over the airstrip before landing
    approach_xy_zeta: float = 1.2
    approach_xy_ki: float = 30.0
    land_radius: float = 1.5
    land_max_speed: float = 20.0
    approach_timeout_s: float = 3.5
    leg_timeout_s: float = 25.0  # a transit leg longer than this: land now
    # --- dust ------------------------------------------------------------------------
    hopper_mm: float = 470.0  # mm of spraying per load (~2 fields of 6 x 38 mm)
    swath: float = 2.0  # half width of the settled dust (mm)
    dose: float = 1.0  # dust dose per pass at the swath centre (cells saturate at 1)
    settle_speed: float = 45.0  # mm/s, the dust's fall speed (drift = wind x fall time)
    regrow_s: float = 100.0  # a fully dusted cell is green again after this long
    dusted_frac: float = 0.8  # a field counts as dusted when this share of cells is >= 0.5
    n_puffs: int = 70
    puff_every_s: float = 0.02
    puff_life_s: float = 1.3
    dust_every_s: float = 0.02  # dust / crop / particle update period
    # --- ground work ---------------------------------------------------------------------
    refill_s: float = 2.0
    settle_s: float = 0.3
    retry_after_s: float = 0.6
    max_retry_tilt_deg: float = 30.0
    strip_xy: tuple = (0.0, 0.0)
    # --- wind (physical: MuJoCo medium velocity) ------------------------------------------
    wind_max: float = 18.0  # mm/s breeze
    wind_period_s: float = 29.0
    gust_mean_s: float = 22.0  # a gust on average this often
    gust_speed: float = 14.0  # mm/s added at the gust's peak
    gust_s: float = 2.5
    # --- presentation ---------------------------------------------------------------------
    caption_s: float = 1.6
    cam_distance: float = 27.0
    cam_elevation: float = -26.0
    cam_yaw_tau_s: float = 1.0


@register_job
class CropDusterJob(EternalJob):
    name = "crop_duster"
    znear = 0.3
    zfar = 400.0
    title = "CROP DUSTER FLY"
    tagline = "the fly crop-dusts fields forever"
    work_label = "fields dusted"
    config_cls = CropDusterConfig
    needs_flight = True
    required_names = (P + "strip", P + "cell0", P + "puff0")

    def __init__(self, cfg: CropDusterConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.fields = [tuple(float(v) for v in f) for f in c.fields]
        self.n_seg = int(round(c.field_len / c.seg_len))
        self.n_cells = len(self.fields) * c.n_rows * self.n_seg
        self.phase = "ground"
        self.t_phase = 0.0
        self.field = 0
        self.row = 0  # rows done in this field
        self.row_order: list[int] = []
        self.row_dir = 1.0
        self.hopper = c.hopper_mm
        # counters
        self.fields_dusted = 0
        self.rows = 0
        self.refills = 0
        self.gusts = 0
        self.dust_used = 0.0
        self.distance = 0.0
        self.flight_time = 0.0
        self.n_takeoffs = 0
        self.n_landings = 0
        self.n_crash = 0
        self.n_off_strip = 0
        self.n_flight_aborts = 0
        self.per_field = [0] * len(self.fields)
        self.caption = ""
        self.caption_sub = ""
        self._caption_until = -1.0
        self._caption_rgb = (255, 255, 255)
        # guidance state
        self._v = 0.0
        self._t_leg0 = None
        self._last_xy = None
        self._t_dist = 0.0
        self._cam_yaw = math.pi / 2
        self._cam_dist = c.cam_distance
        self._cam_t = None
        self._wind = np.zeros(2)
        self._gust = None  # (t0, direction unit)
        self._next_gust = 0.0
        self._upright_since = None
        self._touched = False
        self._spraying = False
        self._t_dust = None
        self._t_puff = 0.0
        self._rotor = 0.0
        self._after_takeoff = "climb"
        self._land_prev = None
        self._xint = 0.0
        self._homing_cruise = False
        self._hold_xy = np.zeros(2)
        self._braked = False
        self._turn_x = 0.0
        self._t_brake = 0.0
        self._wp = np.zeros(2)
        self._wp_yaw = 0.0

    # ------------------------------------------------------------ build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)  # flat, no auto hits, flight fly (needs_flight)
        f = app_cfg.flight
        f.hover_s = None  # the job decides when to land
        f.turn_rate = self.cfg.turn_rate

    def field_box(self, f: int) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) of field ``f``."""
        c = self.cfg
        cx, cy = self.fields[f]
        hy = c.n_rows * c.row_gap / 2
        return cx - c.field_len / 2, cx + c.field_len / 2, cy - hy, cy + hy

    def row_y(self, f: int, r: int) -> float:
        x0, x1, y0, y1 = self.field_box(f)
        return y0 + self.cfg.row_gap * (r + 0.5)

    def cell_layout(self) -> np.ndarray:
        """(n_cells, 4): field, row, x, y of every crop cell."""
        c = self.cfg
        out = []
        for f in range(len(self.fields)):
            x0, x1, _, _ = self.field_box(f)
            for r in range(c.n_rows):
                for s in range(self.n_seg):
                    out.append((f, r, x0 + c.seg_len * (s + 0.5), self.row_y(f, r)))
        return np.array(out)

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._materials(spec)

        def mesh(name, md, pos, material, quat=(1.0, 0.0, 0.0, 0.0), body=None):
            A.add_mesh(spec, P + name + "_mesh", md)
            (body or wb).add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + name + "_mesh",
                                  pos=tuple(pos), quat=tuple(quat), material=P + material, **vis)

        # the fields: soil, crop cells (visual; the material shows the dust stage), fences
        A.add_mesh(spec, P + "crop_mesh", A.crop_segment_mesh(c.seg_len, 1.1, 0.55, c.seed))
        for f in range(len(self.fields)):
            x0, x1, y0, y1 = self.field_box(f)
            mesh(f"soil{f}", A.box_mesh(((x1 - x0) / 2 + 0.6, (y1 - y0) / 2 + 0.6, 0.02), 0.08),
                 ((x0 + x1) / 2, (y0 + y1) / 2, 0.02), "soil")
        for k, (f, r, x, y) in enumerate(self.cell_layout()):  # (one contiguous block of geoms)
            wb.add_geom(name=f"{P}cell{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "crop_mesh",
                        pos=(float(x), float(y), 0.04), material=P + "crop0", **vis)
        for f in range(len(self.fields)):
            x0, x1, y0, y1 = self.field_box(f)
            m = 1.2
            for i, (p0, p1) in enumerate((((x0 - m, y0 - m), (x1 + m, y0 - m)), ((x1 + m, y0 - m), (x1 + m, y1 + m)),
                                          ((x1 + m, y1 + m), (x0 - m, y1 + m)), ((x0 - m, y1 + m), (x0 - m, y0 - m)))):
                mesh(f"fence{f}_{i}", A.fence_mesh(p0, p1), (0, 0, 0), "wood")
        # the airstrip, windsock, hangar, dust hopper (the refill station)
        sx, sy = c.strip_xy
        wb.add_geom(name=P + "strip", type=mj.mjtGeom.mjGEOM_BOX, size=(10.0, 2.2, 0.02), pos=(sx, sy, 0.02),
                    material=P + "strip", **vis)
        mesh("hangar_shell", A.hangar_meshes(5.0, 4.4, 2.2)["shell"], (sx - 9.0, sy - 4.5, 0), "hangar")
        mesh("hopper_tank", A.hopper_meshes()["tank"], (sx + 6.0, sy + 4.0, 0), "hopper")
        mesh("hopper_legs", A.hopper_meshes()["legs"], (sx + 6.0, sy + 4.0, 0), "steel")
        mesh("hopper_chute", A.hopper_meshes()["chute"], (sx + 6.0, sy + 4.0, 0), "steel")
        mesh("windsock_pole", A.windsock_meshes()["pole"], (sx + 9.0, sy - 3.5, 0), "steel")
        sock = wb.add_body(name=P + "windsock", mocap=True, pos=(sx + 9.0, sy - 3.5, 3.2))
        for i in range(4):
            sock.add_geom(name=P + f"sock{i}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.28 - 0.045 * i, 0.2, 0),
                          pos=(0.2 + 0.4 * i, 0, 0), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                          material=P + ("sock_r" if i % 2 == 0 else "sock_w"), **vis)
        mesh("strip_sign", A.panel_mesh(3.6, 1.5, 0.05), (sx - 8.0, sy + 3.6, 1.9), "strip_sign",
             A.panel_quat((0, -1, 0)))
        mesh("strip_sign_post", A.polyline_tube(np.array([[0, 0, 0], [0, 0, 1.2]]), 0.07, 6),
             (sx - 8.0, sy + 3.62, 0), "wood")
        # the farmyard either side of the strip: barn, silo, windpump
        bm = A.barn_meshes()
        for nm, mat in (("walls", "barn"), ("roof", "roof"), ("trim", "white")):
            mesh(f"barn_{nm}", bm[nm], (-48.0, 2.0, 0), mat)
        sm = A.silo_meshes()
        for nm, mat in (("body", "silo"), ("dome", "steel"), ("bands", "steel")):
            mesh(f"silo_{nm}", sm[nm], (-40.0, 0.0, 0), mat)
        mesh("windpump", A.windpump_meshes()["tower"], (46.0, 4.0, 0), "steel")
        rot = wb.add_body(name=P + "rotor", mocap=True, pos=(46.0, 4.0 - 0.4, 11.8))
        rm = A.windpump_rotor()
        for nm, mat in (("blades", "white"), ("hub", "steel"), ("tail", "roof")):
            mesh(f"rotor_{nm}", rm[nm], (0, 0, 0), mat, body=rot)
        # trees and hedgerows round the farm
        rng = np.random.default_rng(c.seed + 4)
        spots = [(-54, 20), (54, 28), (-52, 48), (52, 52), (-30, -22), (30, -20), (0, 64), (-26, 64), (26, 64),
                 (0, -26), (-12, 66), (14, 67)]
        for i, (x, y) in enumerate(spots):
            tm = A.tree_meshes(rng.uniform(5.5, 8.5), seed=i)
            mesh(f"tree{i}_trunk", tm["trunk"], (x, y, 0), "bark")
            mesh(f"tree{i}_crown", tm["crown"], (x, y, 0), "leaves")
        for i, (x, y) in enumerate(((0.0, 34.0), (0.0, 22.0), (0.0, 46.0))):
            mesh(f"hedge{i}", A.shrub_mesh((0, 0), 1.3, seed=10 + i, n=9), (x, y, 0), "hedge")
        # the dust puffs (a fixed pool of mocap particles, visual)
        for i in range(c.n_puffs):
            b = wb.add_body(name=f"{P}puff{i}", mocap=True, pos=(0.0, 0.0, -20.0))
            b.add_geom(name=f"{P}puff{i}_g", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.35, 0.35, 0.22),
                       rgba=(0.97, 0.93, 0.74, 0.0), material=P + "dust", **vis)
        # sky backdrop and light
        mesh("sky", A.panel_mesh(420.0, 130.0, 0.5), (0.0, 150.0, 35.0), "sky", A.panel_quat((0, -1, 0)))
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.40, 0.62, 0.92)
            sky.rgb2 = (0.88, 0.92, 0.97)
        spec.visual.headlight.ambient = (0.36, 0.36, 0.34)
        spec.visual.headlight.diffuse = (0.34, 0.34, 0.32)
        spec.visual.headlight.specular = (0.06, 0.06, 0.06)
        sun = np.array([-40.0, -60.0, 120.0])
        wb.add_light(name=P + "sun", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=tuple(sun), dir=tuple(-sun),
                     diffuse=(0.52, 0.50, 0.44), specular=(0.15, 0.15, 0.14), castshadow=False)
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(40, 60, 60),
                     dir=(-0.4, -0.5, -1.0), diffuse=(0.16, 0.18, 0.22), specular=(0.02, 0.02, 0.02),
                     castshadow=False)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        grid = spec.material("grid")
        if grid is not None:  # the meadow, one texture tile per 16 mm
            A.add_texture(spec, P + "tex_meadow", A.meadow_texture(c.seed))
            grid.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_meadow"
            grid.texrepeat = [2000.0 / 16.0, 2000.0 / 16.0]
            grid.rgba = (1.0, 1.0, 1.0, 1.0)
            grid.reflectance = 0.0
            grid.specular = 0.05
        A.add_texture(spec, P + "tex_soil", A.soil_texture(c.seed))
        T(spec, P + "soil", P + "tex_soil", rgba=(1, 1, 1, 1), specular=0.05)
        for k in range(A.N_STAGES):
            A.add_texture(spec, P + f"tex_crop{k}", A.crop_texture(k, c.seed))
            T(spec, P + f"crop{k}", P + f"tex_crop{k}", rgba=(1, 1, 1, 1), specular=0.1 + 0.05 * k,
              emission=0.04 * k)
        A.add_texture(spec, P + "tex_strip", A.strip_texture())
        T(spec, P + "strip", P + "tex_strip", rgba=(1, 1, 1, 1), specular=0.05, texrepeat=(1, 1))
        A.add_texture(spec, P + "tex_barn", A.barn_texture(c.seed))
        T(spec, P + "barn", P + "tex_barn", rgba=(1, 1, 1, 1), specular=0.1)
        A.add_texture(spec, P + "tex_sign", A.sign_texture("FLY-BY FARMS", "AIRSTRIP - DUST REFILL"))
        T(spec, P + "strip_sign", P + "tex_sign", rgba=(1, 1, 1, 1), emission=0.2)
        A.add_texture(spec, P + "tex_sky", A.sky_texture(c.seed))
        T(spec, P + "sky", P + "tex_sky", rgba=(1, 1, 1, 1), emission=0.85)
        M = spec.add_material
        for nm, rgba, sp in (("wood", (0.55, 0.40, 0.25, 1), 0.1), ("roof", (0.40, 0.40, 0.43, 1), 0.3),
                             ("white", (0.95, 0.95, 0.93, 1), 0.2), ("silo", (0.75, 0.76, 0.74, 1), 0.6),
                             ("steel", (0.58, 0.60, 0.63, 1), 0.7), ("hangar", (0.50, 0.53, 0.56, 1), 0.5),
                             ("hopper", (0.92, 0.72, 0.12, 1), 0.5), ("bark", (0.40, 0.29, 0.19, 1), 0.05),
                             ("leaves", (0.24, 0.46, 0.20, 1), 0.1), ("hedge", (0.20, 0.40, 0.18, 1), 0.1),
                             ("sock_r", (0.95, 0.35, 0.10, 1), 0.1), ("sock_w", (0.97, 0.97, 0.95, 1), 0.1)):
            M(name=P + nm, rgba=rgba, specular=sp, shininess=0.5)
        M(name=P + "dust", rgba=(1, 1, 1, 1), emission=0.2, specular=0.0)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        c = self.cfg
        self.fm = self.session.flight
        if self.fm is None:
            raise RuntimeError("crop_duster needs the flight fly (cfg.flight.enabled)")
        self.fm.ground_height_fn = lambda x, y: 0.0
        self.fm.listeners.append(self._on_flight_event)
        lay = self.cell_layout()
        self.cell_f = lay[:, 0].astype(int)
        self.cell_r = lay[:, 1].astype(int)
        self.cell_xy = lay[:, 2:4].copy()
        g0 = m.geom(P + "cell0").id
        self.cell_geoms = np.arange(g0, g0 + self.n_cells)
        assert m.geom(f"{P}cell{self.n_cells - 1}").id == self.cell_geoms[-1]
        self.crop_mats = np.array([m.material(P + f"crop{k}").id for k in range(A.N_STAGES)])
        self.dust = np.zeros(self.n_cells)
        self._stage = np.zeros(self.n_cells, int)
        self.puff_mocap = np.array([int(m.body_mocapid[m.body(f"{P}puff{i}").id]) for i in range(c.n_puffs)])
        self.puff_geom = np.array([m.geom(f"{P}puff{i}_g").id for i in range(c.n_puffs)])
        self.puff_pos = np.zeros((c.n_puffs, 3))
        self.puff_pos[:, 2] = -20.0
        self.puff_vel = np.zeros((c.n_puffs, 3))
        self.puff_t0 = np.full(c.n_puffs, -1e9)
        self._puff_next = 0
        self.mocap_sock = int(m.body_mocapid[m.body(P + "windsock").id])
        self.mocap_rotor = int(m.body_mocapid[m.body(P + "rotor").id])
        self._rng = np.random.default_rng(c.seed + 13)
        self._next_gust = self._rng.exponential(c.gust_mean_s) + 6.0
        self._set("ground")
        self._t_ground0 = self.sim.time

    # ------------------------------------------------------------ helpers
    def ground_height(self, x: float, y: float) -> float:
        return 0.0

    def _set(self, phase: str) -> None:
        self.phase = phase
        self.state = phase
        self.t_phase = self.sim.time

    def _flash(self, text: str, sub: str = "", rgb=(255, 255, 255)) -> None:
        self.caption, self.caption_sub, self._caption_rgb = text, sub, rgb
        self._caption_until = self.run_time() + self.cfg.caption_s

    def field_need(self) -> float:
        c = self.cfg
        return c.n_rows * (c.field_len + 2 * c.overrun)

    def field_coverage(self, f: int) -> float:
        sel = self.cell_f == f
        return float(np.mean(self.dust[sel] >= 0.5))

    def _pick_field(self) -> int:
        """The field whose crop has grown back the most (least dust), not the one just done."""
        dust = [float(np.mean(self.dust[self.cell_f == f])) for f in range(len(self.fields))]
        order = np.argsort(dust)
        for f in order:
            if int(f) != self.field or len(self.fields) == 1:
                return int(f)
        return int(order[0])

    def _plan_field(self, f: int) -> None:
        """Rows from the side nearest the fly, the first pass from the nearer end."""
        c = self.cfg
        self.field = f
        self.row = 0
        p = self.sim.com()[:2]
        x0, x1, y0, y1 = self.field_box(f)
        rows = list(range(c.n_rows))
        if abs(p[1] - y1) < abs(p[1] - y0):
            rows = rows[::-1]
        self.row_order = rows
        self.row_dir = 1.0 if abs(p[0] - x0) <= abs(p[0] - x1) else -1.0

    def _row_ends(self) -> tuple[np.ndarray, np.ndarray]:
        """(start, end) of the current pass: from ``lead_in`` before the field edge
        (the fly is up to speed at the edge) to ``overrun`` past the far edge."""
        c = self.cfg
        x0, x1, _, _ = self.field_box(self.field)
        y = self.row_y(self.field, self.row_order[min(self.row, c.n_rows - 1)])
        if self.row_dir > 0:
            return np.array([x0 - c.lead_in, y]), np.array([x1 + c.overrun, y])
        return np.array([x1 + c.lead_in, y]), np.array([x0 - c.overrun, y])

    def _velocity_mode(self, on: bool) -> None:
        fm = self.fm
        if fm.ctrl is None:
            return
        fm.ctrl.g = replace(fm._gains, xy_zeta=self.cfg.cruise_xy_zeta) if on else fm._gains

    def _com_vel(self) -> np.ndarray:
        ctrl = self.fm.ctrl
        if ctrl is None or not ctrl.log:
            return self.sim.thorax_linvel()
        rows = list(ctrl.log)[-23:]
        return np.mean([r[4:7] for r in rows], axis=0)

    def _hold(self, xy, alt: float, yaw: float) -> tuple[float, float]:
        """Position loop on (xy, alt) with the heading goal ``yaw``: (position error,
        yaw error)."""
        fm = self.fm
        ctrl = fm.ctrl
        fm._speed = 0.0
        if fm.state != "hovering":
            fm.state = "hovering"
        # descend at most descend_rate (a fast drop overshoots below the set point)
        dt = self.cfg.update_every_steps * self.sim.timestep
        fm._clearance = max(alt, fm._clearance - self.cfg.descend_rate * dt)
        fm._yaw_goal = yaw
        ctrl.target.pos = np.r_[np.asarray(xy, float), ctrl.target.pos[2]]
        e = float(np.hypot(*(np.asarray(xy) - self.sim.com()[:2])))
        return e, abs(wrap_angle(yaw - ctrl.target.yaw))

    def _fly_along(self, dt: float, aim_xy, v_des: float, alt: float, heading: float | None = None) -> None:
        """Velocity control: heading toward ``aim_xy`` (or ``heading``), speed ramped to
        ``v_des`` (0 while the heading error is large), altitude ``alt``."""
        c, fm = self.cfg, self.fm
        ctrl = fm.ctrl
        com = self.sim.com()
        rel = np.asarray(aim_xy, float) - com[:2]
        if heading is None:
            if float(np.hypot(*rel)) > 0.5:
                fm._yaw_goal = math.atan2(rel[1], rel[0])
        else:
            fm._yaw_goal = heading
        err = abs(wrap_angle(fm._yaw_goal - ctrl.target.yaw))
        if err > math.radians(c.turn_first_deg):
            v_des = 0.0
        if v_des > self._v:
            self._v = min(self._v + c.accel * dt, v_des)
        else:
            self._v = max(v_des, self._v - 2 * c.decel * dt)
        ctrl.target.pos = np.r_[com[:2], ctrl.target.pos[2]]
        ctrl.xy_int[:] = 0.0
        fm._clearance = alt
        fm._speed = self._v
        want = "forward" if self._v > 0 else "hovering"
        if fm.state != want:
            fm.state = want

    # ------------------------------------------------------------ events
    def _on_flight_event(self, ev) -> None:
        if ev.kind == "crash":
            self.n_crash += 1
            self._flash("CRASH LANDING!", "", (255, 110, 90))
            self._spraying = False
            self._set("down")
        elif ev.kind == "abort":
            self.n_flight_aborts += 1
            if self.phase not in ("ground", "refill", "down", "wait"):
                self._set("down")
        elif ev.kind == "land":
            self.n_landings += 1

    def on_reset(self) -> None:
        # explicit reset: the fly is back on the airstrip (the FlyGym spawn)
        self._v = 0.0
        self._last_xy = None
        self._spraying = False
        self.sim.model.opt.wind[:] = 0.0
        self._gust = None
        self.puff_t0[:] = -1e9
        self._set("ground")
        self._t_ground0 = self.sim.time
        self._t_dust = None

    # ------------------------------------------------------------ the loop
    def update(self) -> None:
        sim, fm, c = self.sim, self.fm, self.cfg
        dt = c.update_every_steps * sim.timestep
        t = sim.time - self.t_phase
        self._update_wind()
        if fm.airborne:
            self.flight_time += dt
            if abs(sim.time - self._t_dist) >= 0.01:
                self._t_dist = sim.time
                xy = sim.com()[:2]
                if self._last_xy is not None:
                    self.distance += float(np.hypot(*(xy - self._last_xy)))
                self._last_xy = xy
        else:
            self._last_xy = None
        ph = self.phase
        self._spraying = False
        if ph == "ground":  # on the strip after a (re)spawn: take off
            self.steering.set(None, 0.0)
            if t >= 0.4 and fm.state == "walking" and sim.tilt_deg() < c.max_retry_tilt_deg:
                if self.hopper < self.field_need():  # (respawned on the strip: refill first)
                    self._set("refill")
                    self._flash("REFILLING", "the hopper tops up the dust", (255, 230, 120))
                    return
                self._goto_field(self._pick_field() if self.fields_dusted or self.rows else 0)
                self._takeoff()
        elif ph == "takeoff":
            if fm.state in ("hovering", "forward") and fm.ctrl is not None:
                fm._clearance = c.transit_alt
                self._approach_gains()  # stiffer position loop for every hold (wind)
                self._set(self._after_takeoff)
            elif fm.state == "walking" and t > 0.25:
                self._set("wait")
        elif ph in ("climb", "transit", "lineup", "pass", "turn", "home", "approach"):
            if fm.ctrl is not None and fm.state in ("hovering", "forward"):
                getattr(self, "_ph_" + ph)(dt, t)
        elif ph == "landing":
            if fm.state == "touchdown" and not self._touched:
                self._touched = True
                if fm.ctrl is not None:
                    fm.ctrl.xy_int[:] = 0.0
                    fm.ctrl.target.pos = np.r_[sim.com()[:2], fm.ctrl.target.pos[2]]
            if fm.state == "walking":
                self._set("landed")
        elif ph == "landed":
            self.steering.set(None, 0.0)
            if t >= c.settle_s and fm.state == "walking":
                if sim.tilt_deg() > 60.0:
                    self.n_crash += 1
                    self._flash("CRASH LANDING!", "", (255, 110, 90))
                    self._set("down")
                else:
                    if float(np.hypot(*(sim.com()[:2] - np.asarray(c.strip_xy)))) > 6.0:
                        self.n_off_strip += 1
                    self._set("refill")
                    self._flash("REFILLING", "the hopper tops up the dust", (255, 230, 120))
        elif ph == "refill":
            self.steering.set(None, 0.0)
            if t >= c.refill_s:
                self.hopper = c.hopper_mm
                self.refills += 1
                self.last_work_rt = self.run_time()
                self._goto_field(self._pick_field())
                self._takeoff()
        elif ph in ("down", "wait"):
            self._maybe_retry(t)
        now = sim.time
        if self._t_dust is None or now < self._t_dust:
            self._t_dust = now
        if now - self._t_dust >= c.dust_every_s:
            self._dust_step(now - self._t_dust)
            self._t_dust = now

    def _goto_field(self, f: int) -> None:
        self._plan_field(f)
        self._after_takeoff = "climb"

    def _takeoff(self) -> None:
        msg = self.fm.takeoff(source="job")
        if msg.startswith("[flight] take-off"):
            self.n_takeoffs += 1
            self._t_leg0 = self.sim.time
            self._v = 0.0
            self._after_takeoff = "climb"
            self._set("takeoff")
        else:
            self._set("wait")

    # --- flight phases -------------------------------------------------------------
    def _ph_climb(self, dt: float, t: float) -> None:
        c, fm = self.cfg, self.fm
        start, _ = self._row_ends() if self._target_is_field() else (np.asarray(c.strip_xy, float), None)
        self._hold(self.sim.com()[:2], c.transit_alt, self._bearing(start))
        err = abs(wrap_angle(fm._yaw_goal - fm.ctrl.target.yaw))
        if self.sim.com()[2] > c.transit_alt - 2.0 and err < math.radians(c.turn_first_deg):
            self._velocity_mode(True)
            self._v = 0.0
            self._set("transit" if self._target_is_field() else "home")

    def _target_is_field(self) -> bool:
        return self._after_takeoff != "home_now"

    def _bearing(self, xy) -> float:
        rel = np.asarray(xy, float) - self.sim.com()[:2]
        return math.atan2(rel[1], rel[0]) if float(np.hypot(*rel)) > 0.5 else self.fm.ctrl.target.yaw

    def _cruise_to(self, dt: float, t: float, dest, alt: float, next_phase: str) -> None:
        c = self.cfg
        com = self.sim.com()
        d = float(np.hypot(*(np.asarray(dest) - com[:2])))
        v_des = min(c.transit_speed, math.sqrt(2.0 * c.decel * max(d - c.handover_mm, 0.0)))
        self._fly_along(dt, dest, v_des, alt)
        if self._t_leg0 is not None and self.sim.time - self._t_leg0 > c.leg_timeout_s:
            self._velocity_mode(False)
            self._land()
            return
        if d <= c.handover_mm + 0.2 or (d < 3 * c.handover_mm and self._v < 1.0 and t > 0.3):
            self._v = 0.0
            self._velocity_mode(False)
            self.fm.ctrl.xy_int[:] = 0.0
            self._set(next_phase)

    def _ph_transit(self, dt: float, t: float) -> None:
        start, end = self._row_ends()
        self._cruise_to(dt, t, start, self.cfg.transit_alt, "lineup")

    def _ph_lineup(self, dt: float, t: float) -> None:
        """Over the pass start: descend to the pass altitude, turn to the row heading."""
        c = self.cfg
        start, end = self._row_ends()
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        e, ye = self._hold(start, c.pass_alt, yaw)
        dz = abs(self.sim.com()[2] - c.pass_alt)
        if (e < c.lineup_tol and ye < math.radians(c.lineup_yaw_deg) and dz < 0.6) or t > c.lineup_timeout_s:
            self._start_pass()

    def _start_pass(self) -> None:
        self._velocity_mode(True)
        self._v = 0.0
        self._xint = 0.0
        self._set("pass")

    def _ph_pass(self, dt: float, t: float) -> None:
        """A dusting pass: velocity control along the row at the pass altitude, heading
        = pure pursuit onto the row line plus an integral on the cross-track error
        (the crosswind); the dust is on over the field."""
        c = self.cfg
        start, end = self._row_ends()
        com = self.sim.com()
        u = float(np.sign(end[0] - start[0]))
        togo = float((end[0] - com[0]) * u)
        err = float(com[1] - end[1])
        self._xint = float(np.clip(self._xint + c.xtrack_ki * err * dt, -3.0, 3.0))
        aim = np.array([com[0] + u * c.lookahead, end[1] - self._xint])
        self._fly_along(dt, aim, c.pass_speed, c.pass_alt)
        x0, x1, _, _ = self.field_box(self.field)
        if x0 - 0.2 <= com[0] <= x1 + 0.2 and self.hopper > 0 and self._v > 5.0:
            self._spraying = True
        if togo < 0.0:
            self._row_done()

    def _row_done(self) -> None:
        c = self.cfg
        self.rows += 1
        self.last_work_rt = self.run_time()
        self.row += 1
        self._braked = False
        if self.row >= c.n_rows:
            cov = self.field_coverage(self.field)
            if cov >= c.dusted_frac:
                self.fields_dusted += 1
                self.per_field[self.field] += 1
                self.add_work(1)
                self._flash(f"FIELD {self.field + 1} DUSTED", f"{100 * cov:.0f}% of the crop treated", (255, 240, 150))
            else:
                self._flash(f"FIELD {self.field + 1}: {100 * cov:.0f}% ONLY", "the wind took the rest", (255, 200, 120))
            self.session.log_event("field_dusted", job=self.name, field=self.field + 1, coverage=round(cov, 3))
            self._t_leg0 = self.sim.time
            self._v = 0.0
            self._velocity_mode(False)
            self.fm.ctrl.xy_int[:] = 0.0
            self._homing_cruise = False
            if self.hopper < self.field_need():
                self._after_takeoff = "home_now"
                self._set("home")
            else:
                self._plan_field(self._pick_field())
                self._after_takeoff = "climb"
                self._set("climb")
            return
        self.row_dir = -self.row_dir
        self._set("turn")

    def _ph_turn(self, dt: float, t: float) -> None:
        """Pull-up turn at the row end: brake along the row while climbing ``pullup``,
        then hover-turn 180 deg while the position loop slides over to the next row
        (where the braking ended) and drops back to the pass altitude, then the next
        pass."""
        c = self.cfg
        com = self.sim.com()
        up = c.pass_alt + c.pullup
        if not self._braked:
            fwd = self.fm.ctrl.target.yaw
            aim = com[:2] + 5.0 * np.array([math.cos(fwd), math.sin(fwd)])
            self._fly_along(dt, aim, 0.0, up, heading=fwd)
            if float(np.hypot(*self._com_vel()[:2])) < 10.0 or t > 0.6:
                self._braked = True
                self._v = 0.0
                self._velocity_mode(False)
                self.fm.ctrl.xy_int[:] = 0.0
                x0, x1, _, _ = self.field_box(self.field)
                self._turn_x = float(np.clip(com[0], x0 - c.lead_in - 4.0, x1 + c.lead_in + 4.0))
                self._t_brake = t
            return
        start, end = self._row_ends()
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        e, ye = self._hold((self._turn_x, start[1]), c.pass_alt, yaw)
        dz = abs(com[2] - c.pass_alt)
        if ((e < c.lineup_tol and ye < math.radians(c.lineup_yaw_deg) and dz < 0.5)
                or t - self._t_brake > c.lineup_timeout_s):
            self._start_pass()

    def _ph_home(self, dt: float, t: float) -> None:
        c = self.cfg
        com = self.sim.com()
        strip = np.asarray(c.strip_xy, float)
        if not self._homing_cruise:
            # climb out of the field and turn toward the strip first
            self._hold(self._hold_xy if t > 0 else com[:2], c.transit_alt, self._bearing(strip))
            if t < 0.05:
                self._hold_xy = com[:2].copy()
            err = abs(wrap_angle(self.fm._yaw_goal - self.fm.ctrl.target.yaw))
            if (com[2] > c.transit_alt - 2.0 and err < math.radians(c.turn_first_deg)) or t > 2.0:
                self._homing_cruise = True
                self._velocity_mode(True)
                self._v = 0.0
            return
        self._cruise_to(dt, t, strip, c.transit_alt, "approach")
        if self.phase == "approach":
            self._approach_gains()

    def _ph_approach(self, dt: float, t: float) -> None:
        c = self.cfg
        strip = np.asarray(c.strip_xy, float)
        e, _ = self._hold(strip, c.approach_alt, 0.0)
        vxy = float(np.hypot(*self._com_vel()[:2]))
        dz = abs(self.sim.com()[2] - c.approach_alt)
        if (e < c.land_radius and vxy < c.land_max_speed and dz < 0.8) or t > c.approach_timeout_s:
            self._land()

    def _approach_gains(self) -> None:
        fm, c = self.fm, self.cfg
        g = replace(fm._gains, xy_wn=c.approach_xy_wn, xy_zeta=c.approach_xy_zeta, xy_ki=c.approach_xy_ki)
        fm._gains = g
        if fm.ctrl is not None:
            fm.ctrl.g = g

    def _land(self) -> None:
        msg = self.fm.land(source="job")
        self._touched = False
        if msg == "[flight] landing":
            self._set("landing")

    def _maybe_retry(self, t: float) -> None:
        sim, fm, c = self.sim, self.fm, self.cfg
        if fm.state != "walking" or self.fly_down():
            self._upright_since = None
            return
        if sim.tilt_deg() > c.max_retry_tilt_deg:
            self._upright_since = None
            return
        if self._upright_since is None:
            self._upright_since = sim.time
        if sim.time - self._upright_since >= c.retry_after_s and t >= c.retry_after_s:
            self._upright_since = None
            if self.hopper < self.field_need():
                self._set("refill")  # (landed near enough: refill here)
                return
            self._goto_field(self.field if self.row < self.cfg.n_rows else self._pick_field())
            self._takeoff()

    # ------------------------------------------------------------ dust / crop / props
    def _dust_step(self, dt: float) -> None:
        c = self.cfg
        sim = self.sim
        d, m = sim.data, sim.model
        now = sim.time
        rt = self.run_time()
        com = sim.com()
        w3 = np.array([self._wind[0], self._wind[1], 0.0])
        # emit puffs while spraying
        if self._spraying:
            vel = self._com_vel()
            n = int(dt / c.puff_every_s + 0.5)
            for _ in range(max(n, 1)):
                i = self._puff_next
                self._puff_next = (i + 1) % c.n_puffs
                jitter = self._rng.normal(0, 0.12, 3)
                self.puff_pos[i] = com + np.array([0, 0, -0.7]) + jitter - vel * self._rng.uniform(0, dt)
                self.puff_vel[i] = 0.25 * vel + np.array([0, 0, -c.settle_speed * 0.5]) + self._rng.normal(0, 3, 3)
                self.puff_t0[i] = now
            used = float(np.hypot(*vel[:2])) * dt
            self.hopper = max(0.0, self.hopper - used)
            self.dust_used += used
            # the plume settles downwind of the track (wind x fall time); the cells whose
            # centres the settling point passed since the last update get the dose
            fall = max(com[2] - 0.5, 0.0) / c.settle_speed
            land = com[:2] + self._wind * fall
            prev = self._land_prev if self._land_prev is not None else land - vel[:2] * dt
            self._land_prev = land.copy()
            lo, hi = min(prev[0], land[0]), max(prev[0], land[0])
            ymid = 0.5 * (prev[1] + land[1])
            lat = np.abs(self.cell_xy[:, 1] - ymid)
            sel = (self.cell_xy[:, 0] >= lo) & (self.cell_xy[:, 0] < hi) & (lat < c.swath)
            if np.any(sel):
                k = np.flatnonzero(sel)
                w = 1.0 - (lat[k] / c.swath) ** 2
                self.dust[k] = np.minimum(1.0, self.dust[k] + c.dose * w)
        else:
            self._land_prev = None
        # particles: drift with the wind, slow down, settle, fade
        age = now - self.puff_t0
        live = (age >= 0.0) & (age < c.puff_life_s)
        if np.any(live):
            P_ = self.puff_pos
            V_ = self.puff_vel
            k = np.flatnonzero(live)
            drag = math.exp(-dt / 0.15)
            V_[k] = w3 + (V_[k] - w3) * drag
            V_[k, 2] = np.where(P_[k, 2] > 0.25, -c.settle_speed * (1 - drag) + V_[k, 2] * drag, 0.0)
            P_[k] += V_[k] * dt
            P_[k, 2] = np.maximum(P_[k, 2], 0.2)
            a = age[k] / c.puff_life_s
            size = 0.4 + 1.3 * np.sqrt(a)
            m.geom_size[self.puff_geom[k], 0] = size
            m.geom_size[self.puff_geom[k], 1] = size
            m.geom_size[self.puff_geom[k], 2] = size * 0.55
            m.geom_rbound[self.puff_geom[k]] = size
            m.geom_rgba[self.puff_geom[k], 3] = 0.5 * (1 - a) ** 1.3
        dead = ~live
        self.puff_pos[dead, 2] = -20.0
        m.geom_rgba[self.puff_geom[dead], 3] = 0.0
        d.mocap_pos[self.puff_mocap] = self.puff_pos
        # the crop grows back
        self.dust = np.maximum(0.0, self.dust - dt / c.regrow_s)
        stage = np.minimum((self.dust * (A.N_STAGES - 1) + 0.5).astype(int), A.N_STAGES - 1)
        ch = np.flatnonzero(stage != self._stage)
        if len(ch):
            m.geom_matid[self.cell_geoms[ch]] = self.crop_mats[stage[ch]]
            self._stage[ch] = stage[ch]
        # windsock and the windpump's rotor
        w = self._wind
        yaw = math.atan2(w[1], w[0])
        droop = 1.2 * (1.0 - min(1.0, float(np.hypot(*w)) / 30.0))
        d.mocap_quat[self.mocap_sock] = _qmul(quat_axis_angle((0, 0, 1), yaw), quat_axis_angle((0, 1, 0), droop))
        self._rotor += dt * (0.5 + 0.12 * float(np.hypot(*w)))
        d.mocap_quat[self.mocap_rotor] = _qmul(quat_axis_angle((0, 0, 1), -math.pi / 2),
                                               quat_axis_angle((1, 0, 0), self._rotor))  # faces -y

    def _update_wind(self) -> None:
        c = self.cfg
        rt = self.run_time()
        if c.wind_max <= 0 and c.gust_speed <= 0:
            return
        s = rt * 2 * math.pi / c.wind_period_s
        mag = c.wind_max * (0.55 + 0.45 * math.sin(0.61 * s + 1.0))
        ang = 0.9 + 0.8 * math.sin(0.37 * s) + 0.3 * math.sin(1.3 * s)
        w = np.array([mag * math.cos(ang), mag * math.sin(ang)])
        if c.gust_speed > 0:
            if self._gust is None and rt >= self._next_gust:
                a = self._rng.uniform(-math.pi, math.pi)
                self._gust = (rt, np.array([math.cos(a), math.sin(a)]))
                self.gusts += 1
                self._flash("WIND GUST!", f"+{c.gust_speed:.0f} mm/s", (200, 230, 255))
                self.session.log_event("wind_gust", job=self.name, n=self.gusts)
            if self._gust is not None:
                u = (rt - self._gust[0]) / c.gust_s
                if u >= 1.0 or u < 0:
                    self._gust = None
                    self._next_gust = rt + self._rng.exponential(c.gust_mean_s) + 4.0
                else:
                    w = w + c.gust_speed * math.sin(math.pi * u) ** 2 * self._gust[1]
        self._wind[:] = w
        self.sim.model.opt.wind[:2] = w

    # ------------------------------------------------------------ camera / HUD
    def camera_target(self) -> np.ndarray:
        p = self.sim.com()
        if self.phase in ("pass", "turn", "lineup"):
            cx, cy = self.fields[self.field]
            return 0.55 * p + 0.45 * np.array([cx, cy, 0.0])
        return p + np.array([0.0, 0.0, 1.0])

    def camera_preset(self) -> CameraPreset:
        c = self.cfg
        fm = self.fm
        now = self.sim.time
        working = self.phase in ("pass", "turn", "lineup")
        if working:
            goal, dist = math.pi / 2, c.cam_distance + 5.0
        elif fm.ctrl is not None and fm.airborne:
            goal, dist = fm.ctrl.target.yaw, c.cam_distance
        else:
            goal, dist = self._cam_yaw, c.cam_distance - 4.0
        if self._cam_t is None or now < self._cam_t:
            self._cam_yaw, self._cam_dist = goal, dist
        else:
            a = 1.0 - math.exp(-(now - self._cam_t) / max(c.cam_yaw_tau_s, 1e-3))
            self._cam_yaw += a * wrap_angle(goal - self._cam_yaw)
            self._cam_dist += a * (dist - self._cam_dist)
        self._cam_t = now
        return CameraPreset(azimuth=math.degrees(self._cam_yaw), elevation=c.cam_elevation,
                            distance=self._cam_dist, tau_s=0.35)

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """The FIELD DUSTED / REFILLING / WIND GUST caption (a screen overlay)."""
        if t >= self._caption_until or not self.caption:
            return frame
        img = np.ascontiguousarray(frame).copy()
        h, w = img.shape[:2]
        s = w / 720.0
        font = cv2.FONT_HERSHEY_DUPLEX
        for text, scale, y in ((self.caption, 1.25 * s, int(h * 0.15)),
                               (self.caption_sub, 0.6 * s, int(h * 0.15 + 34 * s))):
            if not text:
                continue
            th = max(1, int(round(2.2 * s)))
            (tw, _), _ = cv2.getTextSize(text, font, scale, th)
            org = ((w - tw) // 2, y)
            cv2.putText(img, text, org, font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, text, org, font, scale, self._caption_rgb, th, cv2.LINE_AA)
        return img

    def job_hud_lines(self) -> list[str]:
        fm = self.fm
        c = self.cfg
        return [
            f"field {self.field + 1}  row {min(self.row + 1, c.n_rows)}/{c.n_rows}   [{self.phase}]   "
            f"rows {self.rows}   hopper {100 * self.hopper / c.hopper_mm:3.0f}%   refills {self.refills}",
            f"flight time {self.flight_time:6.1f} s   distance flown {self.distance / 1000:5.2f} m   "
            f"wind {np.hypot(*self._wind):3.0f} mm/s   gusts {self.gusts}",
            "coverage " + "  ".join(f"F{f + 1} {100 * self.field_coverage(f):3.0f}%" for f in range(len(self.fields))),
            fm.hud_line(),
            "real flapping-wing flight + physical wind; dust: visual particles, crop tint by the settled plume",
        ]

    def job_stats(self) -> dict[str, Any]:
        return {
            "fields_dusted": self.fields_dusted, "rows": self.rows, "refills": self.refills,
            "wind_gusts": self.gusts, "distance_mm": round(self.distance, 1),
            "flight_time_s": round(self.flight_time, 2), "dust_used_mm": round(self.dust_used, 1),
            "takeoffs": self.n_takeoffs, "landings": self.n_landings, "crash_landings": self.n_crash,
            "off_strip_landings": self.n_off_strip, "flight_aborts": self.n_flight_aborts,
            "per_field": list(self.per_field),
            "coverage": [round(self.field_coverage(f), 3) for f in range(len(self.fields))],
        }


def _qmul(q1, q2):
    out = np.empty(4)
    mj.mju_mulQuat(out, np.asarray(q1, float), np.asarray(q2, float))
    return out
