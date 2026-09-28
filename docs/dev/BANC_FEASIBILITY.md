# Feasibility: adding a ventral nerve cord (BANC / MANC / FANC) to the brain model

Status: **write-up only, nothing implemented.** Date: 2026-09-28. Roadmap items B10/B11
(docs/dev/ROADMAP.md).

Goal: body touch, nociception and proprioception enter through real VNC sensory
neurons, and descending neurons (DNs) drive real VNC premotor and motor circuits,
instead of today's shortcuts: the hand-picked "nociceptive relay" onto ascending
neurons (docs/STRESS.md) and DN rates mapped onto FlyGym's CPG drive (docs/BRAIN.md,
*Brain → body*).

Labels used below: **[checked]** means I read it from the data files or their metadata
(file listings, small tables I downloaded); **[source]** means a paper, README or data page
states it and I did not check it against the data; **[est]** means my estimate, not measured.

## TL;DR

* **BANC is the dataset to use.** It is one female fly with brain and VNC together:
  188,259 neurons and about 198 M synapses (v3). It was published in *Nature* in 2026, is
  licensed **CC BY 4.0**, and is deposited on Harvard Dataverse as flat Feather/Parquet
  files, so no account is needed. It is annotated with cell types, transmitters, and
  matches to **FlyWire, MANC, FANC and maleCNS**. For our purposes the key fact is that
  it contains **169 VNC sensory neurons annotated `nociception`** (multidendritic) and
  **391 leg motor neurons annotated with their target muscles**. FlyWire has neither.
* **Recommended path: keep the FlyWire v783 brain as it is and add BANC's VNC**, stitched
  through BANC's own FlyWire matches of the DNs and ANs (option **b′** below). A reduced
  VNC (option c) comes first as an offline stage. A full swap to BANC (option a) is
  possible later, but it would invalidate every FlyWire-keyed feature and validation that
  exists today.
* **Speed and memory are not the problem.** Stitching adds about 20 k neurons and
  (est.) 3–6 M connections to an engine that uses 400 MB. The engine's measured cost
  scales with the number of *active* neurons (see *Cost model*). The hard problems are
  elsewhere: (1) a uniform-parameter LIF model with chemical synapses only may not
  produce VNC rhythms or reflexes; (2) motor neurons have to be mapped to NeuroMechFly's
  42 position actuators; (3) a proprioceptive loop needs brain↔body exchange every
  ~5 ms, but the current IPC design has 0.02–0.12 s of lag.
* **Minimum useful first step (2–4 days, about 0.45 GB peak disk, low RAM):** download the
  BANC v888 edgelist and metadata, filter them to a ~50 MB VNC subset, and delete the raw
  files. Then (i) derive the nociceptor → AN → FlyWire-AN relay from the connectome to
  replace the hand-picked `an_walk`/`an_arousal` groups, and (ii) run an offline LIF test
  of the BANC VNC alone (DNg100 drive → leg MN rhythm?). Neither step touches the
  running app.

## 1. Datasets

| | **BANC** (Brain And Nerve Cord) | **MANC** (male adult nerve cord) | **FANC** (female adult nerve cord) | maleCNS (for reference) |
|---|---|---|---|---|
| Coverage | one ♀ fly: brain + optic lobes + neck + whole VNC [source] | ♂ VNC only | ♀ VNC only | one ♂ fly: brain + optic lobes + VNC |
| Paper | Bates, Phelps, Kim, Yang et al., *Nature* 2026, doi [10.1038/s41586-026-10735-w](https://doi.org/10.1038/s41586-026-10735-w) (preprint [2025.07.31.667571](https://www.biorxiv.org/content/10.1101/2025.07.31.667571v3)) | Takemura et al. 2024 eLife ([97769](https://elifesciences.org/reviewed-preprints/97769)); Marin et al. 2024 ([97766](https://elifesciences.org/reviewed-preprints/97766)); Cheong et al. 2024 ([96084](https://elifesciences.org/articles/96084)) | Azevedo, Lesser, Phelps, Mark et al., *Nature* 631:360 (2024) ([link](https://www.nature.com/articles/s41586-024-07389-x)) | Janelia FlyEM, v1.0 June 2026 ([site](https://male-cns.janelia.org/)) |
| Release | CAVE materialization **v888** (snapshot 2026-04-16); Dataverse deposit v3.0, 2026-07-01 [checked] | v1.2.1 (tutorial files); newer on neuPrint | public snapshot with the paper; the latest reconstruction is restricted to community members ([FANC_auto_recon](https://github.com/htem/FANC_auto_recon)) | v1.0 |
| Neurons | **188,259** (meta table); 155,927 in the paper's per-neuron table (Supp. Data 2), 150,852 of them proofread [checked] | 23,650 [source: tutorial docs] | ~14,600 cell bodies [source]; incompletely proofread | ~166 k |
| Synapses / edges | **198.7 M synapses (v3, size ≥ 10)**, 13.5 M neuron→neuron edges; v2 (used in the paper, size ≥ 5): 169 M synapses, 11.5 M edges [source: [data docs](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md)] | ~31 M synapses, 5.3 M edges [source] | ~45 M synapses [source] | 124 M synapses, 25.6 M connections (as counted by [flymsg](https://github.com/gianlucamazza/flymsg)) |
| VNC composition | VNC region (Supp. Data 2): 12,835 intrinsic, 7,591 sensory, 699 motor, 1,849 AN, 514 sensory-ascending, 146 visceral; plus 1,316 DNs in the brain [checked] | similar in scale (DN→MN circuits: Cheong et al.) | leg and wing MN atlas with muscle targets (Azevedo et al.) | – |
| Transmitters | per-neuron and per-synapse predictions over 8 classes (Eckstein et al. 2024 classifier), plus manually verified NTs; VNC: ACh 14,570, GABA 6,579, Glu 2,613, rest < 300 [checked] | predicted_nt per neuron | per-neuron (partial) | per-neuron |
| Cross-matches | columns `fafb_match` (FlyWire root id), `manc_match`, `fanc_match`, `malecns_match`, `hemibrain_match` [checked] | FANC matches in Stürner et al. | MANC matches | – |
| Licence | **CC BY 4.0** (Dataverse) [checked]; tools GPL-3.0 | CC BY [source: Janelia] | not verified | CC BY [source] |
| Download | [Dataverse doi:10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N) (379 files, **536 GB in total**) and the public GCS bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/`; [Codex](https://codex.flywire.ai/banc) | `gs://…/compiled_data/manc_121/`, neuPrint | CAVE (fanc.community) | [download page](https://male-cns.janelia.org/download/) (weights 1.1 GB) |

**BANC files that matter here** (sizes from the Dataverse/GCS listings [checked]):

| file | size | needed for |
|---|---|---|
| `banc_888_edgelist_simple_v3.feather` (pre, post, count, …; 13.5 M rows) | **359 MB** | connectivity (v2: 305 MB) |
| `banc_888_meta.feather` (188 k × ~79 columns: classes, types, NT, matches, body part) | 57.6 MB | neuron table |
| `banc_supplemental_data.zip` (Supp. Data 2 = per-neuron metadata table, 31.7 MB unzipped) | 9.0 MB | a lighter substitute for meta |
| `banc_fafb/manc/fanc_reviewed_matches.csv.gz` | 6.6 / 2.1 / 0.5 MB | reviewed NBLAST matches |
| `banc_888_neurotransmitter_prediction_v2.csv` | 21 MB | optional (meta already has NT) |
| per-synapse parquets (v2 17 GB, v3 19.7 GB), skeletons (17 GB), influence (~287 GB) | – | **not needed** |

Many BANC files are named `*.csv.gz` on Dataverse but are served as plain CSV [checked].

**Tools.** None are strictly needed. pandas + pyarrow, already installed, read the flat
files. [neuprint-python](https://github.com/connectome-neuprint/neuprint-python) (MANC,
maleCNS; token), CAVEclient/[`banc`](https://pypi.org/project/banc/) (live BANC; API key
only for community members), [fafbseg-py](https://github.com/flyconnectome/fafbseg-py)
(FlyWire) and [navis](https://github.com/navis-org/navis) (NBLAST, transforms) are
useful only for fresh matching or morphology, which this plan does not need.

### How DNs and ANs are matched to FlyWire

* **Stürner, Brooks et al., *Nature* 643:158 (2025)** ([doi](https://doi.org/10.1038/s41586-025-08925-z),
  tables in [flyconnectome/2023neckconnective](https://github.com/flyconnectome/2023neckconnective))
  typed every neck neuron in FAFB-FlyWire, FANC and MANC, and matched DNs across the
  datasets. In their supplement [checked]: FAFB has 1,316 DNs in 474 types and MANC
  has 1,328 DNs in 470 types. FAFB ANs carry brain-side names (`AN_AVLP_PVLP_8`),
  while FANC/MANC ANs carry VNC names (`AN09B012`), and the FAFB AN table has no MANC
  column. The reason is that each dataset holds only half of each neck neuron.
* **BANC closes that gap**, because each AN and DN is whole in one animal. In Supp. Data 2
  [checked], 1,304 of 1,316 BANC DNs have a FlyWire match and 1,268 have a MANC match;
  1,713 of 1,849 ANs have a FlyWire match and 1,768 a MANC match; 448 of 516
  sensory-ascending neurons have a FlyWire match. Example: FlyWire `AN_AVLP_PVLP_8` →
  MANC `AN09B012`. The app's `AN_IPS_GNG_7` relay group maps to several MANC types
  (AN04B003, AN06B039, …), so type-level matches are sometimes one-to-many. All the DNs the
  app reads (DNg100, DNg97, DNp09, DNa01/02, MDN, DNp01) are matched in BANC.
* The stricter "reviewed" NBLAST table covers fewer FlyWire neurons (965 of 1,303 DNs,
  876 of 1,750 ANs [checked]). Use the meta columns first and treat the reviewed table as a
  confidence tier.

## 2. Modelling options

### Cost model of the current engine (measured today, M1, single thread)

`scripts/bench_brain.py --seconds 0.5` gave:

| case | spikes/s | active-state neurons | RTF |
|---|---|---|---|
| quiet | 0 | 0 | > 1000 |
| sugar 200 Hz | 15.8 k | 7–11 k | 2.7–3.5 |
| left body mech | 39 k | 8.5–9.6 k | 3.3–4.6 |
| sugar + bitter + body + LC4 | 157 k | 72 k | 0.55–0.58 |

Rule of thumb: **RTF ≈ 40,000 / (active neurons)**. The cost follows the event-driven
active set more than total size or edge count. Memory is about 12 bytes per connection in CSR
(180 MB for 15.1 M) and about 400 MB RSS in total.

### Options

| | (a) swap to BANC whole CNS | (b) FlyWire brain + MANC/FANC VNC | **(b′) FlyWire brain + BANC VNC (recommended)** | (c) reduced VNC |
|---|---|---|---|---|
| What | replace v783 with BANC v888 (all 188 k neurons, or 96 k without optic-lobe intrinsics) | add MANC (♂) or FANC (♀, partial) neurons; merge DN/AN nodes by type | add the BANC VNC-region neurons (~21 k non-neck); DN outputs and AN inputs from the BANC halves, merged onto FlyWire DN/AN nodes through `fafb_match` | only leg premotor/motor + leg sensory + nociceptors + the DN/AN interface; this can be a subset of (b′) |
| Size | 188 k neurons, 13.5 M edges → CSR ≈ 165 MB [est] | +23.6 k, +5.3 M edges → +65 MB | ≈ 160 k neurons, ≈ 18–21 M edges → ≈ 230 MB [est: BANC VNC edge count not measured] | ≈ 5–12 k neurons, 0.5–2 M edges [est]; Pugliese et al. used 4,604 MANC neurons for T1 alone |
| Speed | brain part unchanged; VNC adds its active set (below) | same as b′ | quiet walk with ~5 k VNC active: RTF ≈ 3; heavy transients + walking ≈ 0.5 [est from the rule] | negligible |
| Pros | one animal, one sex, no stitching; nociceptors/MNs native; every neck neuron whole | MANC is the best-studied VNC (DN→MN, CPG papers) | keeps all existing brain work (screens, smell fixes, habituation, fear learning, the Shiu validation); same sex as FlyWire; one annotation system for the VNC | quick to build and inspect; tests the LIF-in-VNC question cheaply |
| Cons | every FlyWire-keyed mapping, CSV and validation must be redone (root ids, `neurons_783.npz`, layout, named sets); BANC's synapse density differs, so the weight must be recalibrated (maleCNS needed ÷1.81 per flymsg); BANC lists problem regions (Supp. Data 10); CC BY vs FlyWire CC BY-NC is fine | male VNC under a female brain (dimorphic DN/ANs: Stürner Supp. files 14–15); FAFB-AN ↔ MANC-AN only through BANC anyway; FANC only partly proofread | still two animals at the neck: DN outputs come from BANC-DN, inputs from FlyWire-DN; the weight scale between datasets needs calibration | leaves out VNC circuits that could matter (e.g. ascending loops) |
| Effort | 3–6 weeks + revalidation [est] | 2–3 weeks | 1–2 weeks for the engine and data, after (c) | 1 week |

**Is uniform-parameter LIF plausible in the VNC?** It is an open question and the main
scientific risk.
* The only published VNC dynamics model, Pugliese et al. 2025/26
  ([bioRxiv 2025.09.12.675944](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v2),
  [code](https://github.com/smpuglie/Pugliese_cpg_2025), no licence file), used a
  **rate** model with jittered per-neuron parameters (τ 20 ± 2 ms, thresholds, firing caps;
  [config](https://github.com/smpuglie/Pugliese_cpg_2025/blob/main/configs/neuron_params/default.yaml)).
  DNg100 stimulation produced leg MN rhythms through a 3-interneuron CPG, which they found
  in four connectomes. No one has shown that Shiu's LIF produces this rhythm.
* Known departures from the uniform model: nonspiking or graded premotor interneurons in
  insects; size-ordered motor neuron recruitment (Azevedo et al. 2020); glutamate as an
  inhibitor (GluCl), which is fine centrally; and **electrical synapses** in the escape
  path (giant fiber → TTMn/PSI through shakB). The GF take-off path therefore needs a
  hand-added gap-junction term.
* Community whole-CNS LIF ports of Shiu to maleCNS exist (e.g.
  [flymsg](https://github.com/gianlucamazza/flymsg), MIT). They validate brain pathways
  (LC4 → GF, sugar → MN9), not walking.
* Plan for this: milestone M1 tests rhythm generation directly. If LIF fails, the
  fallback is a rate-model VNC (Pugliese-style) inside the same process, or a hybrid in
  which the VNC only modulates FlyGym's CPG.

## 3. Body coupling

### Motor neurons → NeuroMechFly

The FlyGym 2.1 NeuroMechFly model has **42 position actuators** (7 DoF per leg, kp 45)
plus adhesion. MuJoCo `MOTOR` and `MUSCLE` actuator types are also available in
`flygym.compose` [checked]. There are no leg muscles. BANC leg MNs by target muscle
[checked; 139/126/126 front/mid/hind]:

| BANC `peripheral_target_type` (n) | joint in NeuroMechFly | plausible mapping |
|---|---|---|
| tibia flexor (30), accessory tibia flexor (71) vs tibia extensor (12) | FTi pitch (`femur-tibia`) | antagonist pair: θ* = θ₀ + g·(r_flex − r_ext) after a ~20–40 ms muscle low-pass |
| trochanter flexor (44), accessory trochanter flexor (18) vs sterno-/tergotrochanter extensor (16/20), trochanter extensor (12) | CTr pitch | same; the mesothoracic tergotrochanter = TTM (jump muscle) → take-off |
| femur reductor (34) | CTr roll | single-sided around a rest angle |
| tarsus depressor (23) vs levator (7); long tendon muscle (48) | TiTa pitch (+ adhesion / grip) | depressor+LTM vs levator |
| sternal anterior/posterior rotator (12/24), promotor (8), remotor/abductor (7), adductor (5) | ThC yaw/pitch/roll | pairs per axis; coxal muscles sit in the thorax |

* **Published precedent:** no paper yet drives NeuroMechFly joints from connectome MNs.
  * NeuroMechFly v2 ([Wang-Chen et al., *Nat Methods* 2024](https://www.nature.com/articles/s41592-024-02497-y))
    uses CPG and preprogrammed steps with DN-like descending signals.
  * Pugliese et al. read MN rates out directly.
  * [Eon Systems, 2026](https://eon.systems/updates/embodied-brain-emulation) drove
    NeuroMechFly from FlyWire LIF DNs through imitation-learned controllers and says
    explicitly that there was no VNC.
  * [Jin et al., arXiv 2602.17997](https://arxiv.org/abs/2602.17997) used the brain graph
    as an RL policy network.
* **Recommendation:** use a staged readout.
  1. First, extract per-leg **phase, frequency and amplitude** from flexor/extensor MN
     alternation, and use them to set the existing CPG oscillators. This is low risk and
     keeps walking stable.
  2. Only later, map MNs to joint targets (the formula in the table) or to MuJoCo
     `MOTOR`/`MUSCLE` actuators. That step is research, with no guarantee that the fly
     walks.

### Body → VNC sensory neurons

BANC VNC sensory neurons by class and body part [checked], and the sim signal each could
come from:

| BANC class (legs F/M/H unless noted) | sim signal (FlyGym/MuJoCo) | notes |
|---|---|---|
| **multidendritic, `nociception`** (169: abdomen 123 + abdominal wall 57, legs 13/11/12, wing 8, thorax 2) | whip/shove impulse above a threshold on that body segment (`WhipHitEvent.details.body`) | the real "pain" entry; replaces the `HIT_RELAY` stand-in |
| bristle (tactile; 973/968/1066 legs, abdomen 777, thorax 298, wing margin 289) | geom contacts per segment (whip, walls, other flies) | Poisson rate ∝ contact force; side and segment from the geom |
| chordotonal (229/248/279; mostly femoral chordotonal organ) | FTi angle (claw), angular velocity (hook, directional), vibration (club) | Mamiya et al. 2018 encoding; the tuning curves are our choice |
| hair plate (coxa 26, legs 56/106/80, prosternal 31) | ThC/CTr angle crossing joint limits | threshold units |
| campaniform sensilla (legs only 53/4/4; haltere 337, wing 158) | per-leg load (tarsus contact force, joint torque) | few leg CS neurons are annotated. Flight: haltere and wing |
| taste bristle, gustatory (289/222/258, wing margin 303) | existing leg taste (docs/TASTE.md) | today these enter at FlyWire's ascending leg GRNs |

## 4. What it unlocks, risks, plan

**Unlocks**
1. Whip "pain" enters at real nociceptors. Which ANs, DNs and OA neurons it reaches becomes
   a connectome prediction instead of our choice.
2. Leg and abdomen touch reaches the brain through real ANs, which gives lateralised
   avoidance and escape routes, plus local VNC reflexes.
3. DNg100/DNb08 → VNC CPG → leg MNs, and MDN → backward stepping, through real wiring.
4. GF → TTMn take-off from the connectome.
5. Grooming DNs (DNg12) → front-leg MNs.

**Risks**

| risk | size | mitigation |
|---|---|---|
| disk (1.7 GB free now) | raw download 0.42 GB (edgelist v3 + meta; 0.37 GB with v2); processed VNC npz ≈ 30–60 MB [est]; full BANC CSR npz ≈ 100–150 MB | filter immediately and delete the raw files; never touch the per-synapse parquets (17–20 GB) |
| RAM (8 GB machine) | reading 3 columns of the edgelist ≈ 0.35 GB peak; engine +≈ 50–100 MB | column-selective `pyarrow.feather.read_table(columns=…)` |
| speed | walking VNC active set of 3–10 k (est) → RTF 1.5–4 alone; with brain transients ≈ 0.4–0.6 | the app already tolerates brain lag; the body runs at 0.5–0.64× |
| **latency** | proprioceptive / MN loops need ≤ 5–10 ms exchange; today's `BrainState` is 0.1 s and lag is 0.02–0.12 s | first stages use slow readouts only (CPG modulation); a closed MN loop needs lockstep brain+physics in one process, or a fast shared-memory channel |
| LIF does not produce rhythms or reflexes | high, unknown | M1 test; rate-model fallback |
| cross-dataset weight scale (FlyWire vs BANC synapse density) | medium | calibrate on matched brain edges (the median BANC/FlyWire count ratio), as flymsg did for maleCNS |
| sex and individual differences at the stitch | low for ♀ BANC; high for MANC | prefer BANC |
| licences | BANC CC BY 4.0, FlyWire CC BY-NC 4.0, MANC CC BY: attribution only; the derived data stays gitignored like `data/brain/` | – |

**Staged plan**

| milestone | content | effort [est] | done when |
|---|---|---|---|
| **M0 (first step)** | `scripts/fetch_banc_vnc.py`: fetch the BANC edgelist v3 and meta, build `data/banc/vnc_888.npz` (VNC + DN/AN/SA neurons, their VNC-side edges, FlyWire/MANC crosswalk), delete the raw files. Derive **nociceptor → AN** weights (1–2 hops), map them to FlyWire AN ids, and offer them as a `noci_relay="banc"` target set in `mapping.named_sets` (still a relay, but connectome-derived) | 2–4 days | report of the ANs, DNs and OA neurons reached; relay targets labelled "BANC-derived" |
| M1 | offline VNC-only LIF (engine unchanged, BANC VNC CSR): DNg100 / DNb08 / MDN / DNp01 / DNg12 Poisson drive → leg MN rates; test for rhythm and left-right / tripod phase; add a GF–TTMn gap-junction term; calibrate the weight scale | 1–2 weeks | an MN rhythm under DNg100, or a documented negative result and a switch to the rate fallback |
| M2 | stitched engine (b′): one CSR = FlyWire + BANC VNC; merge DN/AN nodes; regression shows the brain-only Shiu validation and the existing screens unchanged when the VNC is silent; benchmark RTF | 1–2 weeks | tests pass; RTF table in BRAIN.md |
| M3 | body → VNC sensory: nociceptors, bristles and taste from contacts; hair plates and FeCO from joint states, over a 10 ms channel | 2–3 weeks | whip hit → nociceptors → AN → OA/walk DNs without the relay |
| M4 | VNC → body: MN-derived CPG phase, frequency and amplitude (stable); experimental direct MN → joint / motor actuators | 3–8 weeks, research | the fly walks from DNg100 via VNC MNs (stage 1); stage 2 is open-ended |

**Minimum useful first step (fits this machine):** M0, about 0.45 GB peak disk and a
few minutes of CPU. It adds no running code to the app. It turns "whip → pain" from our
choice into a connectome-derived pathway, and it produces the data file that M1 and M2
build on.

## Sources

* BANC: [htem/BANC-project](https://github.com/htem/BANC-project) (README, `manuscript/print/banc_data_locations.md`),
  [Dataverse doi:10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N) (file list, sizes, licence),
  [data documentation](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md),
  [jasper-tms/the-BANC-fly-connectome](https://github.com/jasper-tms/the-BANC-fly-connectome),
  [FlyWire blog](https://blog.flywire.ai/2025/11/03/the-banc-brain-and-nerve-cord/), Bates et al. 2026 ([doi](https://doi.org/10.1038/s41586-026-10735-w)).
  Supp. Data 2 and 10 and the reviewed-match tables were downloaded (≈ 18 MB) to count classes and matches, then deleted.
* MANC: [manc_data.md](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/manc_data.md), [Janelia MANC page](https://www.janelia.org/project-team/flyem/manc-connectome), Takemura/Marin/Cheong et al. 2024 (eLife links above).
* FANC: Azevedo et al. 2024 ([Nature](https://www.nature.com/articles/s41586-024-07389-x)), [htem/FANC_auto_recon](https://github.com/htem/FANC_auto_recon).
* maleCNS: [download page](https://male-cns.janelia.org/download/), [flymsg](https://github.com/gianlucamazza/flymsg).
* DN/AN matching: Stürner, Brooks et al. 2025 ([Nature](https://doi.org/10.1038/s41586-025-08925-z)), [2023neckconnective](https://github.com/flyconnectome/2023neckconnective).
* VNC dynamics: Pugliese et al. ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v2), [code](https://github.com/smpuglie/Pugliese_cpg_2025)).
* Body: NeuroMechFly v2 ([Nat Methods 2024](https://www.nature.com/articles/s41592-024-02497-y)), [Eon Systems 2026](https://eon.systems/updates/embodied-brain-emulation), [Jin et al. 2026](https://arxiv.org/abs/2602.17997); FlyGym 2.1 `ActuatorType` (installed code).
* Engine cost: `scripts/bench_brain.py --seconds 0.5`, run 2026-09-28 on this M1.
