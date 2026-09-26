# Third-party credits, data licences and references

Fly Simulator's own source code is released under the MIT licence (see
[LICENSE](LICENSE)). It builds on other people's software, data and science. This
file lists them, together with the licence of each part. **Some files in this
repository are derived from FlyWire data and are therefore licensed CC BY-NC 4.0,
not MIT** (see [Files derived from FlyWire data](#files-derived-from-flywire-data)).

Licence information below was checked on 2026-09-26 against the upstream
repositories, the Zenodo record and the installed package metadata.

## Connectome data (downloaded, not included)

`scripts/fetch_brain_data.py` downloads these files into `data/brain/`, which is
gitignored. None of them are part of this repository or of the Python package.

| data | source | licence |
|---|---|---|
| FlyWire whole-brain connectome, public release v783 (connectivity as `Connectivity_783.parquet`, neuron list as `Completeness_783.csv`) | FlyWire Consortium; Dorkenwald et al. 2024; Schlegel et al. 2024. Files taken from [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) @ `91bdd1e` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). [FlyWire guidelines](https://flywire.ai/guidelines): "FlyWire's public release data is made available under license CC BY-NC 4.0". |
| FlyWire neuron annotations (`Supplemental_file1_neuron_annotations.tsv`) | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) v3.1.0 (`8587524`); Schlegel et al. 2024, Matsliah et al. 2024, Berg et al. 2025, Dorkenwald et al. 2024 | The repository has no licence file. Its README asks users to cite the four papers. The annotations are part of the FlyWire public release, so we treat them as CC BY-NC 4.0. |
| Per-neuron neuropil synapse counts (`per_neuron_neuropil_count_pre_783.feather`) | Zenodo record [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866), "FlyWire Whole-brain Connectome Connectivity Data", FlyWire Consortium | The Zenodo record lists **CC BY 4.0**. Because FlyWire's own terms for the public release are CC BY-NC 4.0, we apply the stricter CC BY-NC 4.0 to everything derived from it. |
| Neurotransmitter predictions (columns in the files above) | Eckstein, Bates et al. 2024 | part of the FlyWire release (CC BY-NC 4.0) |

The synapses in the release were detected by Buhmann et al. 2021, using the
synaptic cleft segmentation of Heinrich et al. 2018, in the FAFB electron-microscopy
volume of Zheng et al. 2018.

**FlyWire data may only be used non-commercially.** If you publish work that uses
the brain model, cite Dorkenwald et al. 2024 and Schlegel et al. 2024 (and the other
papers above) as the [FlyWire guidelines](https://flywire.ai/guidelines) ask.

## Files derived from FlyWire data

These files in the repository were produced from FlyWire data. They are licensed
under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution:
the FlyWire Consortium, Dorkenwald et al. 2024, Schlegel et al. 2024), not under MIT:

| file | what it contains |
|---|---|
| `docs/sensory_screen.csv` | Results of the sensory screen (`scripts/screen_sensory.py`, docs/SENSORY_SCREEN.md): FlyWire cell groups and cell-type names and the descending-neuron rates that the brain model computes from the v783 connectome |
| `docs/sensory_screen_touch_oa.csv` | The same for the touch / octopamine screen |
| `fly_simulator/brain_viz/assets/flywire_neuropils_frontal.json` | 2D outlines of the 78 standard neuropils and of the whole brain, in FlyWire (FAFB14.1) space, built by `fly_simulator/brain_viz/build_atlas.py`. Sources: the FlyWire neuropil meshes (`gs://flywire_neuropil_meshes/neuropils/neuropil_mesh_v141_v6`) as packaged in [fafbseg](https://github.com/navis-org/fafbseg-py) (`JFRC2NP.surf.fw.zip`), and the FlyWire whole-brain mesh in [navis-flybrains](https://github.com/navis-org/navis-flybrains) (`FLYWIRE_whole_brain.ply`, made from the FAFB tissue mask by Peter Li, Google). The neuropil shapes themselves are the Ito et al. 2014 neuropils of the JFRC2 template brain (Jenett, Shinomiya and Ito; [VirtualFlyBrain/DrosAdultBRAINdomains](https://github.com/VirtualFlyBrain/DrosAdultBRAINdomains), CC BY 4.0), transformed into FlyWire space. This file ships in the Python package. |
| `fly_simulator/brain_viz/assets/preview_whip.jpg` | A brain-window image drawn with the neuropil atlas above and FlyWire neuron labels (the activity in it is from the mock brain) |
| brain-window images and videos in `docs/media/` | Any picture or video that shows the neuropil atlas, FlyWire cell types or activity of the connectome brain model |

Not derived, for clarity: `docs/real_vision.png` is output of the flyvis model
(MIT, see below) and FlyGym's eyes, with no FlyWire content. The docs and code quote
some FlyWire facts (cell-type names, a few root IDs in `fly_simulator/brain/mapping.py`,
synapse counts in the text). These are used to identify neurons and to describe
results. They are not copies of the dataset. `data/brain/neurons_783.npz`, which
`fetch_brain_data.py` builds locally, is derived data but is never committed.

Note on the packaging of the atlas sources: the `fafbseg` and `navis-flybrains`
Python packages are GPL-3.0. Their code was not used or copied; `build_atlas.py`
only reads the mesh files bundled in their wheels. Neither package states a separate
licence for these mesh files.

## Brain model

* **Shiu et al. 2024, [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)**
  ([MIT](https://github.com/philshiu/Drosophila_brain_model/blob/main/LICENSE),
  Copyright (c) 2023 Philip Shiu and Nico Spiller). The leaky integrate-and-fire
  whole-brain model is reimplemented here in numba (`fly_simulator/brain/engine.py`)
  with Shiu et al.'s equations and default parameters. `fly_simulator/brain/brian2_ref.py`
  restates the Brian2 network of their `model.py` for cross-checking. No files were
  copied. Their licence requires this notice:

  > Permission is hereby granted, free of charge, to any person obtaining a copy of
  > this software and associated documentation files (the "Software"), to deal in the
  > Software without restriction, [...] subject to the following conditions: The above
  > copyright notice and this permission notice shall be included in all copies or
  > substantial portions of the Software.

* The GPL [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) port
  was read for ideas only. No code was copied from it.

## Body, physics and vision

| project | how it is used | licence |
|---|---|---|
| [FlyGym / NeuroMechFly v2](https://github.com/NeLy-EPFL/flygym) (Wang-Chen et al. 2024; `flygym==2.1.0`) | Runtime dependency: fly model, hybrid CPG walking controller, compound eyes | [Apache-2.0](https://github.com/NeLy-EPFL/flygym/blob/main/LICENSE), Copyright 2023-2026 The NeuroMechFly v2 Authors |
| [NeuroMechFly v1](https://github.com/NeLy-EPFL/NeuroMechFly) (Lobato-Rios et al. 2022) | `fly_simulator/actions/data/grooming_front_legs.npz` is **converted from** their DeepFly3D grooming recording (`data/joint_tracking/grooming/fly1/df3d/joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl`). Changes: we re-solved the joint angles for FlyGym 2.1's leg convention and kept the front-leg grooming segment (`fly_simulator/actions/convert_grooming.py`). | [Apache-2.0](https://github.com/NeLy-EPFL/NeuroMechFly/blob/main/LICENSE), Copyright 2021 Neuroengineering Laboratory, EPFL. The converted file remains under Apache-2.0. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) (Todorov et al. 2012; `mujoco 3.9`) | Physics engine and rendering (installed with FlyGym) | Apache-2.0 |
| [flyvis](https://github.com/TuragaLab/flyvis) (Lappalainen et al. 2024; optional `vision` extra) | Connectome-constrained visual system model for `--real-vision`. Pretrained models are downloaded by the user. | [MIT](https://github.com/TuragaLab/flyvis/blob/main/LICENSE), Copyright (c) 2023 Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill, Jakob H. Macke, Srinivas C. Turaga |
| [flybody](https://github.com/TuragaLab/flybody) (Vaxenburg et al. 2025) | Reference only: the flight model uses flybody's published wing stroke plane angle and fluid coefficients (docs/FLIGHT.md). No code or assets were copied. | Apache-2.0 |

## Other runtime dependencies

Installed from PyPI and not included in this repository: NumPy (BSD-3-Clause),
opencv-python (Apache-2.0), numba (BSD-2-Clause) and pyarrow (Apache-2.0) and SciPy
(BSD-3-Clause) for the `brain` extra, Brian2 (CeCILL-2.1) for the `brain-ref` extra,
gymnasium (MIT), Stable-Baselines3 (MIT), PyTorch (BSD-style) and TensorBoard
(Apache-2.0) for the `rl` extra, and pytest (MIT) for development.

## References

The papers that the models, parameters and design choices in this project rely on,
collected from the documents in `docs/`. Page numbers were compiled for this file;
check them against the publisher before formal use.

### Connectome, data and tools

* Berg S, et al. (2025). Sexual dimorphism in the complete connectome of the *Drosophila* male central nervous system. *bioRxiv*. doi:10.1101/2025.10.09.680999
* Buhmann J, et al. (2021). Automatic detection of synaptic partners in a whole-brain *Drosophila* electron microscopy data set. *Nature Methods* 18:771-774. doi:10.1038/s41592-021-01183-7
* Dorkenwald S, et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634:124-138. doi:10.1038/s41586-024-07558-y
* Eckstein N, Bates AS, et al. (2024). Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187:2574-2594. doi:10.1016/j.cell.2024.03.016
* FlyWire Consortium (2024). FlyWire Whole-brain Connectome Connectivity Data (v783). Zenodo. doi:10.5281/zenodo.10676866
* Heinrich L, Funke J, Pape C, Nunez-Iglesias J, Saalfeld S (2018). Synaptic cleft segmentation in non-isotropic volume electron microscopy of the complete *Drosophila* brain. *MICCAI 2018*, LNCS 11071:317-325. doi:10.1007/978-3-030-00934-2_36
* Ito K, et al. (2014). A systematic nomenclature for the insect brain. *Neuron* 81:755-765. doi:10.1016/j.neuron.2013.12.017
* Lappalainen JK, et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634:1132-1140. doi:10.1038/s41586-024-07939-3
* Li F, et al. (2020). The connectome of the adult *Drosophila* mushroom body provides insights into function. *eLife* 9:e62576. doi:10.7554/eLife.62576
* Matsliah A, et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634:166-180. doi:10.1038/s41586-024-07981-1
* Schlegel P, et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634:139-152. doi:10.1038/s41586-024-07686-5
* Shiu PK, et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634:210-219. doi:10.1038/s41586-024-07763-9
* Zheng Z, et al. (2018). A complete electron microscopy volume of the brain of adult *Drosophila melanogaster*. *Cell* 174:730-743. doi:10.1016/j.cell.2018.06.019

### Body and simulation

* Günel S, et al. (2019). DeepFly3D, a deep learning-based approach for 3D limb and appendage tracking in tethered, adult *Drosophila*. *eLife* 8:e48571. doi:10.7554/eLife.48571
* Lobato-Rios V, et al. (2022). NeuroMechFly, a neuromechanical model of adult *Drosophila melanogaster*. *Nature Methods* 19:620-627. doi:10.1038/s41592-022-01466-7
* Todorov E, Erez T, Tassa Y (2012). MuJoCo: a physics engine for model-based control. *IEEE/RSJ IROS 2012*, 5026-5033. doi:10.1109/IROS.2012.6386109
* Vaxenburg R, et al. (2025). Whole-body physics simulation of fruit fly locomotion. *Nature*.
* Wang-Chen S, et al. (2024). NeuroMechFly v2: simulating embodied sensorimotor control in adult *Drosophila*. *Nature Methods* 21:2353-2362. doi:10.1038/s41592-024-02497-y

### Escape, looming and vision

* Ache JM, et al. (2019). Neural basis for looming size and velocity encoding in the *Drosophila* giant fiber escape pathway. *Current Biology* 29:1073-1081. doi:10.1016/j.cub.2019.01.079
* Card G, Dickinson MH (2008). Visually mediated motor planning in the escape response of *Drosophila*. *Current Biology* 18:1300-1307. doi:10.1016/j.cub.2008.07.094
* Cruz TL, Pérez SM, Chiappe ME (2021). Fast tuning of posture control by visual feedback underlies gaze stabilization in walking *Drosophila*. *Current Biology* 31:4596-4607.
* Dickinson MH, Muijres FT (2016). The aerodynamics and control of free flight manoeuvres in *Drosophila*. *Phil. Trans. R. Soc. B* 371:20150388. doi:10.1098/rstb.2015.0388
* Hindmarsh Sten T, Li R, Otopalik A, Ruta V (2021). Sexual arousal gates visual processing during *Drosophila* courtship. *Nature* 595:549-553.
* Klapoetke NC, et al. (2017). Ultra-selective looming detection from radial motion opponency. *Nature* 551:237-241. doi:10.1038/nature24626
* Land MF, Collett TS (1974). Chasing behaviour of houseflies (*Fannia canicularis*). *Journal of Comparative Physiology* 89:331-357.
* Muijres FT, Elzinga MJ, Melis JM, Dickinson MH (2014). Flies evade looming targets by executing rapid visually directed banked turns. *Science* 344:172-177. doi:10.1126/science.1248955
* Ribeiro IMA, et al. (2018). Visual projection neurons mediating directed courtship in *Drosophila*. *Cell* 174:607-621. doi:10.1016/j.cell.2018.06.020
* Van Oosterom A, Strackee J (1983). The solid angle of a plane triangle. *IEEE Transactions on Biomedical Engineering* BME-30:125-126.
* von Reyn CR, et al. (2014). A spike-timing mechanism for action selection. *Nature Neuroscience* 17:962-970. doi:10.1038/nn.3741
* von Reyn CR, et al. (2017). Feature integration drives probabilistic behavior in the *Drosophila* escape response. *Neuron* 94:1190-1204. doi:10.1016/j.neuron.2017.05.036

### Descending neurons, walking and grooming

* Bidaye SS, Machacek C, Wu Y, Dickson BJ (2014). Neuronal control of *Drosophila* walking direction. *Science* 344:97-101. doi:10.1126/science.1249964
* Bidaye SS, et al. (2020). Two brain pathways initiate distinct forward walking programs in *Drosophila*. *Neuron* 108:469-485.
* Chen C-L, et al. (2023). Ascending neurons convey behavioral state to integrative sensory and action selection brain regions. *Nature Neuroscience* 26:682-695.
* Guo L, Zhang N, Simpson JH (2022). Descending neurons coordinate anterior grooming behavior in *Drosophila*. *Current Biology* 32:823-833.
* Namiki S, Dickinson MH, Wong AM, Korff W, Card GM (2018). The functional organization of descending sensory-motor pathways in *Drosophila*. *eLife* 7:e34272. doi:10.7554/eLife.34272
* Rayshubskiy A, et al. (2020). Neural circuit mechanisms for steering control in walking *Drosophila*. *bioRxiv*. doi:10.1101/2020.04.04.024703
* Sapkal N, et al. (2024). Neural circuit mechanisms underlying context-specific halting in *Drosophila*. *Nature* 634:191-200 (cited in the docs as the bioRxiv preprint).
* Yang HH, et al. (2023). Fine-grained descending control of steering in walking *Drosophila*. *bioRxiv*.
* Zacarias R, Namiki S, Card GM, Vasconcelos ML, Moita MA (2018). Speed dependent descending control of freezing behavior in *Drosophila melanogaster*. *Nature Communications* 9:3697.

### Mechanosensation, taste and neuromodulation

* Busch S, Selcho M, Ito K, Tanimoto H (2009). A map of octopaminergic neurons in the *Drosophila* brain. *Journal of Comparative Neurology* 513:643-667.
* Crocker A, Sehgal A (2008). Octopamine regulates sleep in *Drosophila* through protein kinase A-dependent mechanisms. *Journal of Neuroscience* 28:9377-9385.
* Inagaki HK, et al. (2012). Visualizing neuromodulation in vivo: TANGO-mapping of dopamine signaling reveals appetite control of sugar sensing. *Cell* 148:583-595.
* Kamikouchi A, et al. (2009). The neural basis of *Drosophila* gravity-sensing and hearing. *Nature* 458:165-171. doi:10.1038/nature07810
* Roeder T (2005). Tyramine and octopamine: ruling behavior and metabolism. *Annual Review of Entomology* 50:447-477.
* Suver MP, Mamiya A, Dickinson MH (2012). Octopamine neurons mediate flight-induced modulation of visual processing. *Neuron* 75:1005-1015.
* Yorozu S, et al. (2009). Distinct sensory representations of wind and near-field sound in the *Drosophila* brain. *Nature* 458:201-205.

### Plasticity, habituation and learning

* Abbott LF, Varela JA, Sen K, Nelson SB (1997). Synaptic depression and cortical gain control. *Science* 275:220-224.
* Aso Y, et al. (2010). Specific dopaminergic neurons for the formation of labile aversive memory. *Current Biology* 20:1445-1451.
* Aso Y, et al. (2012). Three dopamine pathways induce aversive odor memories with different stability. *PLoS Genetics* 8:e1002768.
* Aso Y, et al. (2014). Mushroom body output neurons encode valence and guide memory-based action selection in *Drosophila*. *eLife* 3:e04580. doi:10.7554/eLife.04580
* Aso Y, Rubin GM (2016). Dopaminergic neurons write and update memories with cell-type-specific rules. *eLife* 5:e16135. doi:10.7554/eLife.16135
* Castellucci V, Kandel ER (1974). A quantal analysis of the synaptic depression underlying habituation of the gill-withdrawal reflex in *Aplysia*. *PNAS* 71:5004-5008.
* Claridge-Chang A, et al. (2009). Writing memories with light-addressable reinforcement circuitry. *Cell* 139:405-415.
* Cohn R, Morantte I, Ruta V (2015). Coordinated and compartmentalized neuromodulation shapes sensory processing in *Drosophila*. *Cell* 163:1742-1755.
* Engel JE, Wu C-F (1996). Altered habituation of an identified escape circuit in *Drosophila* memory mutants. *Journal of Neuroscience* 16:3486-3499.
* Groves PM, Thompson RF (1970). Habituation: a dual-process theory. *Psychological Review* 77:419-450.
* Handler A, et al. (2019). Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning. *Cell* 178:60-75.
* Hige T, Aso Y, Modi MN, Rubin GM, Turner GC (2015). Heterosynaptic plasticity underlies aversive olfactory learning in *Drosophila*. *Neuron* 88:985-998.
* Honegger KS, Campbell RAA, Turner GC (2011). Cellular-resolution population imaging reveals robust sparse coding in the *Drosophila* mushroom body. *Journal of Neuroscience* 31:11772-11785.
* Owald D, et al. (2015). Activity of defined mushroom body output neurons underlies learned olfactory behavior in *Drosophila*. *Neuron* 86:417-427.
* Tsodyks MV, Markram H (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS* 94:719-723.
* Turner GC, Bazhenov M, Laurent G (2008). Olfactory representations by *Drosophila* mushroom body neurons. *Journal of Neurophysiology* 99:734-746.

### LIF parameters (as used by Shiu et al. 2024)

The engine's synaptic time constant, refractory period and synaptic delay come from
Shiu et al. 2024, who cite Jürgensen et al. 2021, Lazar et al. 2021 and Paul et al.
2015 for them (see their Methods).
