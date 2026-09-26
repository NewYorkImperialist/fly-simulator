# FlyGym 2.x API notes (verified against the installed code)

Installed: **flygym 2.1.0** (PyPI, `requires_python >=3.12,<3.15`), **mujoco 3.9.0**,
Python 3.12 (arm64) in `.venv`. Everything below was checked by reading
`.venv/lib/python3.12/site-packages/flygym{,_demo}/` and by running probe scripts;
numbers are from this machine. FlyGym 2.1 dropped PyMJCF/dm_control and uses
**MuJoCo's native `MjSpec` API and raw `mujoco.MjModel`/`MjData`**. Don't use v1 code
(`flygym.mujoco`, `NeuroMechFly(sim_params=...)`, `env.step(action)`,
`physics.named.data`): none of it exists anymore.

## 1. Units (model units, not SI)

| quantity | unit | evidence |
|---|---|---|
| length | **mm** | `BaseFly.SCALE = 1000` ("we simulate length in mm, not m") |
| time | s | `timestep: 1e-4` |
| mass | **g** | `rigging.yaml` thorax mass 0.000307; whole fly `body_subtreemass` = **0.001024 g ≈ 1.02 mg** |
| gravity | mm/s² | `gravity: [0, 0, -9810]` |
| force | **µN** (= g·mm/s²) | fly weight = 0.001024 × 9810 ≈ **10.05 µN** |
| torque | µN·mm | |
| velocity | mm/s, rad/s | walking speed ≈ 14 mm/s |

Measured: a +z force of 3× body weight (~30 µN) on the thorax for 20 ms launches
the fly to z ≈ 5.6 mm (vz ≈ 360 mm/s); 5× weight along +x for 10 ms gives vx ≈ 515 mm/s.
Scale perturbations as multiples of `sim.fly_mass * 9810` (weight).
Adhesion actuators (tarsus5) have `gain=40` → up to ~40 µN (4× body weight) of
pull per leg that is in stance (see `apply_locomotion_action`): upward hits have to
beat adhesion.

## 2. Physics globals (`flygym/assets/model/neuromechfly/mujoco_globals.yaml`)

`timestep 1e-4 s`, `integrator Euler`, `solver Newton`, `iterations 100`,
`noslip_iterations 5`, flags `multiccd` + `energy` on, `compiler fusestatic: true`,
`boundmass 1e-6`. `visual.global.offwidth/offheight = 2048` (offscreen render cap),
`znear 5e-4`, `zfar 250` mm. The world spec inherits these from the fly in
`BaseWorld.add_fly` (`set_mujoco_globals`). Keep the timestep (spec requirement).

## 3. Building fly + world + simulation

```python
from flygym import Simulation                    # flygym.simulation.Simulation
from flygym.anatomy import ContactBodiesPreset, BodySegment, LEGS  # LEGS = ['lf','lm','lh','rf','rm','rh']
from flygym.compose import FlatGroundWorld       # also BlocksTerrainWorld, GappedTerrainWorld, MixedTerrainWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly   # standard walking fly

fly = make_locomotion_fly(name="nmf", add_adhesion=True, colorize=True)
world = FlatGroundWorld()                        # half_size=1000 mm plane "ground_plane"
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),   # position in mm, facing +x
              bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
              add_ground_contact_sensors=True)
sim = Simulation(world)          # compiles, resets to keyframe "neutral"
```

* `make_locomotion_fly` (in `flygym_demo.complex_terrain.common`): `NeuroMechFly`,
  `Skeleton(axis_order=YAW_PITCH_ROLL, joint_preset=LEGS_ONLY)`, joint stiffness 0.05 /
  damping 0.06, passive tarsi (stiffness 7.5), **42 position actuators** (7 per leg,
  `kp=45`, forcerange ±65) on `ActuatedDOFPreset.LEGS_ACTIVE_ONLY`, + 6 adhesion
  actuators. `nq=73, nv=72, nu=48, nbody=69`.
* `add_fly` adds a **free joint named after the fly** (`"nmf"`, qpos[0:7], qvel[0:6])
  on the root body.
* `ContactBodiesPreset`: `ALL`, `LEGS_THORAX_ABDOMEN_HEAD` (default, 55 fly geoms),
  `LEGS_ONLY`, `TIBIA_TARSUS_ONLY` (FlyGym tutorials). Walking trajectories are
  identical for `LEGS_THORAX_ABDOMEN_HEAD` and `TIBIA_TARSUS_ONLY` (only tibia/tarsi
  touch during gait); we use the former so a fallen body collides with the ground.
* `Simulation(world, timestep=None)` attributes: `mj_model`, `mj_data`, `world`,
  `renderer`, `timestep`, `time`. Methods: `step()` (= one `mj_step`), `reset()`
  (`mj_resetDataKeyframe(..., "neutral")`; also zeroes `xfrc_applied`, `ctrl`,
  warnings), `warmup(duration_s=0.05)`, `set_actuator_inputs(fly, ActuatorType.POSITION, arr)`,
  `set_leg_adhesion_states(fly, arr6)`, `get_body_positions(fly)` (n_bodies×3, order
  `fly.get_bodysegs_order()`), `get_body_rotations(fly)` (quats wxyz),
  `get_joint_angles/velocities(fly)`, `get_ground_contact_info(fly)` (see §7),
  `get_bodysegment_contact_forces(fly, segs, ground_only=True)` (world-frame net force
  per segment), `set_renderer(...)`, `render_as_needed()`, `close()`.
* Our wrapper `fly_simulator.simulation.Simulation` builds exactly this and exposes
  `sim.model`, `sim.data`, `sim.fg` (the FlyGym Simulation), `sim.fly`, `sim.world`,
  `sim.thorax_body_id`, `sim.fly_mass`, `sim.fly_dofs` / `sim.other_dofs`,
  `pre_step_hooks`, `post_step_hooks`, `pre_reset_hooks`, `reset_hooks`,
  `world_extensions`, `thorax_position()`, `thorax_quat()`, `thorax_rotmat()`,
  `heading()`, `thorax_linvel()`, `thorax_angvel_local()/world()`, `tilt_deg()`,
  `reset()`, `step(n)`, `check_stability()`. Constructor:
  `Simulation(cfg, world_factory=None, world_extensions=())` (see §14).

## 4. Names in the compiled model

Everything from the fly is prefixed with `"<fly name>/"`.

* **Thorax body: `"nmf/c_thorax"`** (body id 1; root of the fly; the free joint sits
  on it). `mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/c_thorax")`.
* With `fusestatic` and the LEGS_ONLY skeleton, `c_head` has no joint and is **fused
  into the thorax body**: body `nmf/c_head` does not exist, but geom `nmf/c_head` is
  attached to body `nmf/c_thorax` (geoms of the thorax body: `nmf/c_thorax`,
  `nmf/c_head`). Abdomen segments, wings, eyes, antennae stay separate bodies without
  joints (rigidly welded to the thorax; see `m.body_weldid`).
* Leg bodies/geoms: `nmf/{lf,lm,lh,rf,rm,rh}_{coxa,trochanterfemur,tibia,tarsus1..5}`.
  Geom name == body name. Every body has one mesh geom.
* Joints: `nmf/c_thorax-lf_coxa-yaw`, `nmf/lf_coxa-lf_trochanterfemur-pitch`, ...
  Actuators: `<joint name>-position`, adhesion `nmf/lf_tarsus5-adhesion`.
* Ground geom (FlatGroundWorld): `"ground_plane"`, material `"grid"`.
* Keyframe: `"neutral"` (only one). Sensors: `ground_contact_{leg}_leg` (6).

## 5. Pose / velocity access (raw MuJoCo)

* `data.xpos[tid]` (mm), `data.xquat[tid]` (w,x,y,z), `data.xmat[tid].reshape(3,3)`:
  columns = thorax **x forward, y left, z up** in world coordinates.
  `data.xipos[tid]` = body COM.
* Free joint: `qpos[0:3]` position of the thorax frame, `qpos[3:7]` quat;
  **`qvel[0:3]` = linear velocity in the world frame, `qvel[3:6]` = angular
  velocity in the body (local) frame** (MuJoCo free-joint convention).
  `data.cvel` is COM-based spatial velocity (rot first), avoid unless you know it.
* Upright walking (measured): thorax z ≈ 1.10–1.18 mm, tilt (angle of body z to world z)
  2.5–6°, speed ≈ 14.0 mm/s with the default hybrid controller.

## 6. External forces (perturbations)

* `data.xfrc_applied` shape `(nbody, 6)`: `[fx, fy, fz, tx, ty, tz]` in the **world
  frame**, applied at the body's **center of mass** (`xipos`), units µN / µN·mm.
  It persists across steps until you overwrite it; you must zero it when done.
  `sim.reset()` zeroes it.
* Write it in a `pre_step_hook` (runs right before each `mj_step`) and clear it after
  the duration elapsed. Controller writes only `ctrl`, never `xfrc_applied`.

## 7. Contacts and sensors

* **In plain FlyGym, all fly and ground geoms have `contype=0, conaffinity=0`.**
  (PerpetualFly changes the bits, but not the explicit pairs; see §14.) Collisions exist only
  through explicit `<pair>`s that `_GroundContactMixin._set_ground_contact` creates at
  `add_fly` time for every (fly geom in the preset) × (geom in `world.ground_geoms`).
  Pair params come from `flygym.compose.ContactParams()`: friction (1,1,0.02,1e-4,1e-4),
  solref (2e-4, 1), solimp (0.98, 0.99, 1e-5, 0.5, 3), margin 1e-3 mm.
  ⇒ A geom that is not in `world.ground_geoms` *before* `add_fly` is a ghost (no
  collision with the fly). There is no fly self-collision.
* Explicit pairs ignore contype/conaffinity; there is no per-pair enable flag.
  Disable a pooled geom by parking it far away (e.g. z = -100 mm).
* `Simulation.get_ground_contact_info(fly)` → `(found(6), force(6,3), torque, pos,
  normal, tangent)` per leg, order `fly.get_legs_order()`; **only available if the
  world has exactly one ground geom** (otherwise sensors are not created and the
  call raises `TypeError`). For multi-geom terrain use
  `get_bodysegment_contact_forces(fly, [BodySegment("c_thorax"), ...])` or scan
  `data.contact[:data.ncon]` (`geom1`, `geom2`, `mj.mj_contactForce(m, d, i, out6)`,
  force in contact frame; world force = `contact.frame[i].reshape(3,3).T @ out6[:3]`).
* Useful for fall detection: body–ground contacts of geoms `nmf/c_thorax`,
  `nmf/c_head`, `nmf/c_abdomen*` (paired with the default preset).

## 8. Terrain: adding / modifying geoms (probed, see scratch probes)

* Worlds are `MjSpec`s: `world.mjcf_root.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_BOX,
  name=..., size=(hx,hy,hz), pos=..., rgba=..., contype=0, conaffinity=0)`.
  String→enum helpers: `flygym.utils.mjcf.GEOM_TYPES`. Subclass a world (e.g.
  `FlatGroundWorld` or `flygym.compose.world.complex_terrain._ComplexTerrainWorld`
  with `_add_ground_box/_add_ground_plane`) and **append every collidable geom to
  `self.ground_geoms` in `__init__`**, i.e. before `add_fly` creates the pairs.
  Our `Simulation(cfg, world_factory=...)` accepts such a factory.
* **After compile you cannot add geoms to the running model.** Options:
  1. **Pre-allocated geom pool (recommended).** Static worldbody geoms can be
     moved/resized at runtime by writing `model.geom_pos[g]`, `model.geom_quat[g]`,
     `model.geom_size[g]`, `model.geom_rgba[g]` (model, not data). Verified: moving a
     parked box under the fly lifts the fly. **When changing size also update
     `model.geom_rbound[g]` (bounding-sphere radius, box: `norm(size)`) and
     `model.geom_aabb[g] = [0,0,0, *half_extents]`**; with a stale rbound contacts were
     partially missed (thorax z 1.12 instead of 1.33 on a 0.5 mm slab). Geom type and
     mesh cannot change at runtime → keep one pool per shape type.
     Cost: 100 pooled boxes × 36 fly geoms = 3708 pairs → step time unchanged
     (~95 µs), so pools of a few hundred geoms are fine. (`ProceduralTerrain`
     finally uses contype/conaffinity instead of pairs; building ~14k pairs with
     `MjSpec.add_pair` took ~30 s.)
     **BVH gotcha:** dynamic (contype/conaffinity) collisions go through the
     midphase, which uses a bounding-volume hierarchy over the world body's geoms
     (`model.bvh_aabb`, nodes `body_bvhadr[0] .. + body_bvhnum[0]`, leaves
     `bvh_nodeid >= 0`, children `bvh_child`) built from the *compile-time* geom
     positions. Moving a world-body geom without refitting the BVH makes its
     contacts silently disappear (verified: a box moved under a falling body was
     ignored). After moving geoms, recompute each leaf AABB from `geom_pos/quat/aabb`
     and refit internal nodes bottom-up (child index > parent index); see
     `ProceduralTerrain._refit_bvh`. Explicit `<pair>`s bypass the midphase and
     don't need this. Geoms on *moving bodies* (mocap or jointed, e.g. a whip) are
     not affected: their body BVH lives in body-local coordinates.
  2. **Mocap bodies** (`worldbody.add_body(mocap=True)` + geom): move with
     `data.mocap_pos[m.body_mocapid[b]]` / `data.mocap_quat`. Verified working.
     Note `mj_resetDataKeyframe` restores the keyframe's mocap pose.
  3. **Height field**: `spec.add_hfield(name, nrow, ncol, size=(hx, hy, z_top, z_base),
     userdata=[0.0]*(nrow*ncol))` (userdata is required when there's no file) + an
     hfield geom (in `ground_geoms`). `model.hfield_data[adr:adr+nrow*ncol]` (values
     in [0,1], elevation = value × z_top) can be rewritten at runtime and collisions
     follow immediately (verified). Rendering needs a re-upload:
     `mj.mjr_uploadHField(model, renderer._mjr_context, hfield_id)` for each
     `mujoco.Renderer`. hfield–mesh contact generates many contacts (~170 under the fly).
  4. `spec.recompile(model, data)` → new `(model, data)` with state preserved, ~45 ms
     for this model. **But** FlyGym's `Simulation` caches body/geom/actuator ids and
     holds the old model; the renderer holds the old model; world geoms precede fly
     geoms so every fly geom id shifts; new geoms also need `spec.add_pair(...)`
     for contacts. Avoid unless you rebuild everything.
* MuJoCo planes are infinite for collision; `size` only limits drawing. Shifting a
  plane in-plane is physically a no-op (verified bit-identical trajectory);
  `fly_simulator.terrain.GroundRecentering` uses this to keep the flat checkerboard
  under the fly (snapped to texture periods, so pixel-identical).
  **Gotcha (found in the soak pass):** a geom compiled at its body's origin with
  identity orientation gets `model.geom_sameframe = 1`, and `mj_kinematics` then
  copies the body frame into `geom_xpos` and never reads `geom_pos`. The ground plane
  sits at the world origin, so writing `geom_pos` alone did nothing (the
  checkerboard stayed at x = 0 and the fly walked off it after ~1 m).
  `GroundRecentering` clears the flag. Pool geoms are compiled at `PARK_POS`
  (not the origin), so they have `sameframe = 0` already.
* FlyGym's built-in terrain worlds (`BlocksTerrainWorld`, `GappedTerrainWorld`,
  `MixedTerrainWorld`) are finite static layouts (x ∈ [-10, 25] mm) — useful as
  geometry references only (block 1.3 mm, height 0.35 mm, gap 0.3 mm, in the
  hybrid tutorial the fly spawns at z=1.2 on mixed terrain).

## 9. Locomotion controllers (`flygym_demo.complex_terrain`, installed with flygym)

* `PreprogrammedSteps()` – per-leg recorded step splines; `default_pose_by_dof_order(dof_order)`,
  `swing_period[leg]`, `get_joint_angles(leg, phase, magnitude)`.
* `make_tripod_cpg_network(timestep, intrinsic_frequency=12, intrinsic_amplitude=1,
  coupling_strength=10, convergence_coef=20, seed=0)` → `CPGNetwork` (phases random
  from seed).
* `CPGController(cpg_network, preprogrammed_steps, output_dof_order).step()` →
  `LocomotionAction(joint_angles, adhesion_onoff)`.
* `HybridController(timestep, cpg_network=None, preprogrammed_steps, output_dof_order,
  ...)` – CPG + retraction + stumbling reflexes; `.reset(seed=)`,
  `.step(HybridControllerObservation.from_sim(sim, fly_name))`, `.last_info`.
* `HybridTurningController.step(descending_signal[2], obs)` – `[left, right]` sets the
  CPG amplitudes per side (negative reverses that side's frequency). Larger left →
  turn right.
* `apply_locomotion_action(sim, fly_name, action)` writes position targets + adhesion.
* `output_dof_order = fly.get_actuated_jointdofs_order("position")` (42 DoFs).
* Canonical loop (tutorial 4c): `sim.reset(); ctrl.reset(seed=0);
  apply(initial default pose, adhesion all on); sim.warmup();` then every physics step
  `obs → ctrl.step → apply → sim.step()`.
* Performance: reference controller+observation ≈ 390 µs/step vs `mj_step` ≈ 100 µs.
  `fly_simulator.controllers.FastHybrid(Turning)Controller` / `FastObservationBuilder`
  give **bit-identical** actions (tests `test_fast_controller_matches_flygym`,
  `test_fast_turning_controller_is_bitwise_flygym`) at ≈ 45 µs/step; whole loop
  ≈ 0.75× real time headless. See §13.
* Behaviour: the plain hybrid controller settles at a constant ~10° heading offset to
  the left (walks straight but diagonally). PerpetualFly adds a heading-hold P loop
  (`ControllerConfig.heading_gain=1.5`) feeding `HybridTurningController`: y stays
  within ~0.6 mm over 210 mm.

## 10. Rendering

* `flygym.Renderer(mj_model, cameras, camera_res=(h, w), playback_speed, output_fps,
  buffer_frames)` wraps `mujoco.Renderer`, renders only model cameras by name and
  buffers frames for `save_video()`/`show_in_notebook()`. Attach with
  `sim.set_renderer(...)`; `sim.render_as_needed()` renders based on sim time.
* `fly.add_tracking_camera(name="trackcam", mode="track", pos_offset=(-0.5,-7.5,5),
  rotation=Rotation3D("xyaxes",(1,0,0,0,0.6,0.8)), fovy=30)` is called before
  `world.add_fly` in all tutorials (do the same); the camera is a child of the thorax, offset in the thorax frame;
  compiled name `"nmf/trackcam"`. "track" follows rigidly → jittery with gait bob.
* PerpetualFly instead uses a free `mujoco.MjvCamera` (`mjCAMERA_FREE`) with smoothed
  `lookat` / azimuth, rendered through `mujoco.Renderer.update_scene(data, camera=cam)`
  (`fly_simulator/rendering.py`). Free-camera convention: camera position =
  `lookat - distance * (cos el cos az, cos el sin az, sin el)`; the free camera uses
  `model.vis.global_.fovy`. Rear three-quarter view of a fly heading ψ: az = ψ−45°.
  Smoothing is adaptive (`CameraConfig`): the look-at time constant shrinks with the
  lag (`tau / (1 + (lag / catchup_distance)^2)`), the lag is capped at `max_lag`, the
  camera zooms out when the fly is > `zoom_above` mm above the local ground, and
  the heading is frozen while tilt > `freeze_heading_tilt_deg` (a fly on its back
  has a meaningless heading). With a fixed 0.15 s low-pass, a level-4 hit (~2.5 m/s)
  would leave the frame within a few frames.
* Floor reflections: the checker material has `reflectance` 0.2, and MuJoCo draws the
  mirrored scene when the scene flag `mjRND_REFLECTION` is set (default).
  `renderer.scene.flags[mjtRndFlag.mjRND_REFLECTION] = 0` turns it off.
  `update_scene` / `mjv_updateScene` doesn't reset `scene.flags`, so setting it once
  is enough. Measured draw time at 960×640 on normal terrain: 10.6–11.2 ms with
  reflections, 5.5–6.4 ms without (`RenderConfig.reflections`, `--no-reflections`).
* `flygym.launch_interactive_viewer(m, d)` = blocking `mujoco.viewer.launch`
  (MuJoCo steps the physics itself — no controller). Not useful for us.

## 11. Live display on macOS (decision)

* `mujoco.viewer.launch_passive` **raises `RuntimeError` under plain `python` on
  macOS** ("requires that the Python script be run under `mjpython`"). It works under
  `.venv/bin/mjpython` (verified: opens, `key_callback` receives GLFW key codes).
  Downsides: must launch via `mjpython`, and the viewer's built-in bindings (space =
  pause, ESC, many letters toggling render flags) fire in addition to `key_callback`,
  colliding with our R/B/S/G/F/P/X/SPACE controls.
* **Chosen: OpenCV window** (`fly_simulator/interaction/viewer.py`): offscreen
  `mujoco.Renderer` frame → `cv2.imshow`; keyboard via `cv2.waitKeyEx` (runs on the
  main thread under plain `python`; verified window opens and runs). Key names from
  `fly_simulator.interaction.decode_key` ("space", "left", "right", "up", "down",
  "esc", letters). macOS arrow codes: 63234 left, 63235 right, 63232 up, 63233 down.
  Window close detected with `cv2.getWindowProperty(title, WND_PROP_VISIBLE) < 1`.
  Only key *presses* are delivered (no key-up / held-key state).
* Cost per displayed frame on this Mac: render 960×640 ≈ 14 ms (≈ 13 ms even at
  640×427: not fill-bound), imshow + HUD ≈ 1 ms, `waitKeyEx(1)` ≈ 12–15 ms (macOS
  event loop; `cv2.pollKey()` is no faster). The live window therefore steps physics
  in a worker thread (§13): windowed ≈ 0.72× real time at 30 fps, headless ≈ 0.75×.
* Retina: OpenCV's Cocoa backend maps one image pixel to one *physical* pixel, so a
  960×640 frame showed at 480×320 pt on this Mac (backing scale 2, 1440×900 pt).
  `fly_simulator/display.py` reads the backing scale through CoreGraphics (ctypes) and
  `LiveViewer` upscales ×2 (cv2.resize linear, ~3 ms), then draws the HUD at the
  upscaled resolution. The brain window uses the same helper. At ×2, `imshow` +
  `waitKeyEx(1)` takes ~21 ms instead of ~14 ms, so the window runs ~24 fps instead of
  ~30. Override with `FLY_SIMULATOR_FLY_SCALE` / `RenderConfig.display_scale`.

## 12. Instability

* MuJoCo by default **auto-resets `MjData`** (`mj_resetData`) when qpos/qvel/qacc
  become bad (`mjWARN_BADQACC/BADQPOS/BADQVEL`), which would silently teleport the fly
  to the origin. PerpetualFly sets `model.opt.disableflags |= mjDSBL_AUTORESET` and
  `Simulation.check_stability()` raises `SimulationInstabilityError` on non-finite
  qpos/qvel/qacc, |qvel| > `SimConfig.max_abs_qvel` (1e6) on the **fly's** DoFs
  (`sim.fly_dofs` = DoFs whose body is in the thorax subtree), |qvel| >
  `SimConfig.max_abs_qvel_other` (1e9) on any other DoF (bodies from world
  extensions, e.g. a fast whip tip), thorax z < -50 mm, or any of those warnings
  (`data.warning[w].number`). The diagnostics label each offending DoF fly/non-fly.

## 13. Performance

Measured on this Mac (M-series, mujoco 3.9, Python 3.12), default config, seed 0.
Old and new code were timed side by side in the same process, so machine load
affects both equally.

| | before | after |
|---|---|---|
| `Simulation.step` | 240 µs/step (0.41×) | 134 µs/step (**0.75×**) |
| of which raw `mj_step` | ≈ 77–95 µs | same (the floor: max ≈ 1.0–1.3×) |
| observation build | 47 µs | 7 µs |
| controller step | 110 µs | 40 µs |
| live window (960×640) | 0.23×, 15 fps | **0.72×, 30 fps** |

**The trajectory is bit-for-bit unchanged**: the qpos/qvel after 50k steps match the
pre-optimisation code exactly (`np.array_equal`). The same holds between headless,
threaded-window and `--no-thread` runs.

What changed:
* `FastObservationBuilder`: boolean geom-id lookup tables instead of 4× `np.isin`.
* `FastHybridController`: the per-leg Python loop is vectorised over the 6 legs. The
  6 scipy `CubicSpline` calls are replaced by `_VectorizedSteps`, which repeats
  scipy's periodic wrap, interval search and `evaluate_poly1` in the same float-op
  order, so it is bit-identical. `np.interp` stays numpy's own function, called
  once per left/right leg pair: a hand-written interp is *not* bit-identical
  because numpy's C code uses FMA on arm64.
* `FastHybridTurningController.step`: no `np.repeat`/`asarray`. `apply()` writes
  `data.ctrl` through precomputed index arrays.
* `Simulation.step`: bound locals. `check_stability` has a cheap fast path, and
  diagnostics are only built on failure. With **no** hooks registered, the steps
  between controller updates go to MuJoCo as `mj_step(m, d, nstep)` (bit-identical;
  rarely used because the flat world always has the GroundRecentering hook).
  Hooks still run every physics step.
* MuJoCo flags: FlyGym's globals turn on `mjENBL_ENERGY`, but nothing reads
  `data.energy`, so `SimConfig.compute_energy=False` clears it. That saves 1–2 % with a
  bit-identical trajectory. Disabling sensors (`mjDSBL_SENSOR`) saved nothing, so they
  stay on. (`multiccd` is on: in MuJoCo 3.x it is a *disable* bit, which is why
  `enableflags` shows only ENERGY.) Solver/Jacobian settings change the numerics and
  were not touched. MuJoCo threading gives nothing here (one fly = one island).
* Live window (`app.py` + `fly_simulator/physics_thread.py`): physics runs in a
  worker thread (`RenderConfig.threaded_physics=True`, chunks of
  `thread_chunk_steps=50`). The main thread renders and polls keys at
  `RenderConfig.target_fps=30`, paced by the wall clock. `mj_step`, `Renderer.render`
  and `cv2.waitKeyEx` all release the GIL. Two details matter:
  (1) a "gate" lock as a turnstile, because `threading.Lock` is unfair and the
  display starved at 0.7 fps; (2) `sys.setswitchinterval(0.001)` while the worker
  runs, since the default 5 ms GIL switch interval cut physics from 0.71× to 0.49×.
  Anything touching `sim` from the main thread (key handlers, perturbation calls,
  terrain spawning) must go inside `with runner.locked():`. Hooks run in the worker
  thread. `FrameRenderer.update_scene()` needs the lock; `draw()` doesn't.
  `--no-thread` switches back to the single-threaded loop, which is also paced by
  wall clock: ≈ 0.31× at 20 fps. `--record` and `--headless` always run
  single-threaded, and recording keeps one frame per `render_every_steps`.

Options measured but **off by default**:
* `SimConfig.control_every_steps=k` runs the controller every k steps and holds
  `ctrl`; the CPG/reflex integrators use dt = k·timestep, so gait frequency is kept.
  Over 5 s: k=2 → 0.74×, k=5 → 0.84×, k=10 → 0.91× (k=1: 0.57× in that loaded run).
  Speed (14.0 mm/s), height and tilt are unchanged, but the path deviates by
  ≈ 0.2 mm from k=1. A 1e-9 mm initial perturbation stays at 1e-9 mm, so this is a
  real, if small, behavioural change and not numerical noise. Keep k=1 when
  comparing runs or controllers.
* Reflections: turning off `mjRND_REFLECTION` cuts a frame from 14 to 8 ms. That
  matters only for single-threaded recording, so it was not added as an option.


## 14. World extensions, contact bits, reset hooks (for adding bodies such as a whip)

**`world_extensions`.** `Simulation(cfg, world_factory=None, world_extensions=[ext, ...])`
calls every `ext(world)` after the world is built and **before**
`world.add_fly(...)`. `world` is a FlyGym `BaseWorld`; its `MjSpec` is
`world.mjcf_root` (e.g. `world.mjcf_root.worldbody.add_body(...)`, `.add_geom`,
`.add_freejoint()`, `add_joint`, `world.mjcf_root.add_actuator(...)`). Why before:
`add_fly` → `_rebuild_neutral_keyframe` compiles the spec and writes the "neutral"
keyframe sized to the model **at that moment**. Bodies/joints added after it would
make the keyframe's qpos too short.
The app accepts the same argument: `fly_simulator.app.run(cfg, world_extensions=[...])`
and `Session(cfg, world_extensions=[...])`.

**Keyframe gotcha (handled).** FlyGym fills the "neutral" keyframe only for the fly's
joints and the fly's free joint. Every other joint gets qpos = 0, so a free joint
from an extension would be reset to the world origin with a zero quaternion.
`Simulation` patches the compiled keyframe: for non-fly joints, `key_qpos = qpos0`
(the pose written in the spec), and for mocap bodies `key_mpos/mquat` = the spec pose.
`sim.reset()` therefore puts extension bodies back where the extension created them.
Extension actuators get `ctrl = 0` in the keyframe. (`key_mpos` / `key_mquat` are
flat arrays of shape `(nkey, 3*nmocap)` / `(nkey, 4*nmocap)`; the patch indexes them
accordingly. It used to crash for mocap bodies until the whip exercised it.)

**Contact bits.** Two geoms collide dynamically if `(contype1 & conaffinity2) ||
(contype2 & conaffinity1)`. Constants are in `fly_simulator.terrain` (`FLY_BIT = 8`,
`TERRAIN_BIT = 16`):

| geom | contype | conaffinity | notes |
|---|---|---|---|
| fly contact geoms (`FlyConfig.ground_contact` preset: legs, thorax, head, abdomen) | `FLY_BIT` | 0 | no fly self-collision |
| base plane `ground_plane` | `TERRAIN_BIT` | 0 | touches the fly only through FlyGym's explicit pairs |
| procedural terrain pool | `TERRAIN_BIT` | `FLY_BIT` | priority 1 + FlyGym `ContactParams` |
| **your geom, hits the fly only** | 0 | `FLY_BIT` | |
| **your geom, hits fly + ground + terrain** | 0 | `FLY_BIT \| TERRAIN_BIT` | |
| your geom, hits nothing | 0 | 0 | visual only |

The flat default world (`Simulation(cfg)` without `world_factory`) is a
`TerrainWorld` with no pooled geoms. It uses the same scheme, and its trajectory is
bit-identical to FlyGym's `FlatGroundWorld` (checked against a saved 50k-step
reference). Fly geoms not in the contact preset (wings, antennae, eyes...) keep
contype 0 and are never hit. For your geoms, set `priority=1` and copy
`flygym.compose.ContactParams()` (`condim=3`, `friction=(sliding, torsional,
rolling)`, `solref=cp.get_solref_tuple()`, `solimp=cp.get_solimp_tuple()`,
`margin=cp.margin`) if you want the same stiff contact as the ground. MuJoCo's
defaults (`solref` 0.02 s) are far too soft for a 1 mg fly: a 0.4 mm ball passed
through the thorax. Give bodies explicit masses: the default density of 1000 in
model units is 1000 g/mm³. See `tests/test_app.py::_add_ball` for a
working free body: a 0.1 mg ball lands on the thorax, rolls off and rests on the
plane.
Keep extension geoms out of the fly's spawn volume: at t=0 the thorax is ~2.1 mm
above the ground (spawn frame 0.8 mm + 1.3 mm) and it settles to ~1.15 mm during the
50 ms warmup. A geom overlapping the fly at t=0 is shot away.

**Moving extension bodies:** jointed bodies are simulated. Mocap bodies are moved
with `data.mocap_pos/mocap_quat` in a `pre_step_hook` (and are restored from the
keyframe on reset). Forces on them: `data.xfrc_applied[body_id]` (world frame, at the
COM, µN). Anything touching `sim` from the main thread in window mode must be inside
`with runner.locked():` (§13).

**Driving a jointed chain from a mocap body:** don't parent the chain to the mocap
body. Mocap bodies have no velocity, so children would be carried kinematically (no
lag, no momentum). Instead weld a free body to the mocap body
(`spec.add_equality(type=mjEQ_WELD, objtype=mjOBJ_BODY, name1=mocap, name2=body)`,
`eq.data = [0,0,0, 0,0,0, 1,0,0,0, 1]` = anchor, identity relpose, torquescale;
`eq.solref=(4e-4, 1)`) and hang the chain from that body. **Never teleport the mocap
target**: a jump of a few mm in one step makes the weld yank the chain hard enough
to go NaN. Rate-limit the target instead (see `fly_simulator/interaction/whip.py`,
`Whip._track`). The physical whip is the worked example; see docs/WHIP.md.

**Stability:** extension DoFs are checked for NaN and against
`SimConfig.max_abs_qvel_other` (1e9) instead of the fly limit (1e6).

**Hooks:** `pre_step_hooks` / `post_step_hooks` run around every `mj_step`.
`pre_reset_hooks` run at the top of `sim.reset()`, before FlyGym's keyframe reset
(the procedural terrain rebuilds its layout there). `reset_hooks` run after the reset and the
warmup. `Simulation.reset()` is: pre_reset_hooks → `mj_resetDataKeyframe("neutral")`
→ controller reset + neutral pose → 50 ms warmup → stability check → reset_hooks.

## 15. App wiring (`fly_simulator.app.Session`)

`Session(cfg, log=None, world_extensions=())` builds `ProceduralTerrain` (from
`cfg.terrain.difficulty/seed/weights`; `"flat"` is an all-flat preset that still has the
spawn pool) → `Simulation(world_factory=terrain.build_world, world_extensions=...)`
→ `terrain.attach` → `install_perturbation(sim, cfg.perturbation, cfg.auto_perturb)`
(the auto perturber is always installed, `enabled` from config, so A toggles it) →
`FallDetector(ground_height_fn=terrain.ground_height_at)` → `RunMetrics` →
`RunLogger`. If `cfg.whip.enabled` (default), a `Whip` is built first
(`Whip.extension` is prepended to the world extensions, then `whip.attach(sim)`) and
passed to `install_perturbation(..., whip=, mode=cfg.session.hit_mode)`: the key
handler routes SPACE / arrows / U to `whip.crack(side, level)` in whip mode and to
`Perturbation.hit` in shove mode; H toggles `controls.mode`. The `AutoPerturber`'s
`perturbation` is replaced by a `HitRouter`, which maps the drawn push direction
to the whip side that pushes that way. Whip results (`WhipHitEvent`, via
`whip.listeners`, in the physics thread) go to `metrics.record_hit` with
`magnitude = mean contact force`, so `HitRecord.impulse` is the measured impulse. In
events.csv they are logged as `whip` rows (the Session swaps `logger.log_hit` in
`metrics.hit_listeners` for a dispatcher). Misses are logged as `whip_miss` and not
counted as hits. `metrics/contacts.py` ignores contacts with `whip/...` geoms (not
terrain), so the fall detector doesn't see a whip strike as body-ground contact.
Perturbation `HitEvent`s go to `metrics.record_hit(ev, time=ev.sim_time,
magnitude=ev.magnitude_uN, duration=ev.duration_s, direction=ev.direction_name)`, and
from there through `metrics.hit_listeners` to `events.csv`. `session.reset(source)`
logs `<source>_reset` and calls `sim.reset()` (counted in `RunMetrics.n_resets`).
`session.after_physics()` (once per physics chunk) prints the fall hint and does the
optional auto reset. Quick-win additions: `session.step_difficulty(±1)` calls
`ProceduralTerrain.set_difficulty(name)`. That swaps the `TerrainGenerator` and
regenerates the loaded chunks from two ahead of the fly's chunk onward, with no model
rebuild. `session.auto_hits_allowed()` is the `AutoPerturber.gate`: hits are skipped
while FALLEN / RECOVERING and for `auto_perturb_resume_after_s` after a "recovered"
event. Screenshots and recordings (`fly_simulator/media.py`) run on the main thread
outside the physics lock. In threaded mode, the events.csv rows they write take the
lock. `ProceduralTerrain.ground_height_at(x, y)` ray-casts
(`mju_rayGeom`) against the pool geoms whose bounding sphere covers (x, y), ~20 µs.
