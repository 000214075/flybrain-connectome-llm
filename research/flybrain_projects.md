# Drosophila brain / connectome projects runnable locally

Target hardware: Windows 11 + WSL2 Ubuntu, 31 GB RAM, ~50 GB free disk, **AMD RX 7900 XTX (ROCm, not CUDA)**.
All sizes below are from live `Content-Length`/GitHub-API checks unless flagged `UNCONFIRMED`.
Anything not directly fetched or API-verified is marked `UNCONFIRMED`.

**Hardware caveat up front:** every GPU path in the fly-simulation ecosystem (Brian2CUDA, NEST GPU, GeNN/PyGeNN,
MuJoCo-Warp, NVIDIA-container FlyBrainLab) is CUDA-only. Nothing below ships a ROCm backend, so on this box the
realistic execution mode is **CPU (WSL2, 31 GB RAM)** or **PyTorch-ROCm** if you install a ROCm wheel manually
(ROCm-on-WSL2 for `gfx1100` is not officially supported → `UNCONFIRMED` whether it works). CPU paths are the
ones I sized for a "few GB / few hours" budget.

---

## 1. Connectome datasets (the actual wiring data)

### 1.1 FlyWire / FAFB female adult brain, release 783 — **the canonical release**

| | |
|---|---|
| Neurons | 139,255 (3,732,460 "connections" per Codex FAFB v783 tile) |
| License | **CC-BY-4.0** (Zenodo record license field, verified via API) |
| Paper | Dorkenwald et al., *Nature* 2024, doi:10.1038/s41586-024-07558-y |

**Connectivity files — Zenodo record 10676866** (`https://zenodo.org/records/10676866`, DOI `10.5281/zenodo.10676866`).
Exact sizes verified from the Zenodo REST API (`https://zenodo.org/api/records/10676866`; the HTML page is
Cloudflare-gated, the API is not):

| File | Size | Notes |
|---|---|---|
| `flywire_synapses_783.feather` | **9,492,998,242 B ≈ 9.5 GB** | all ~130M synapses, pre/post root IDs, NT probabilities, neuropil, xyz |
| `proofread_connections_783.feather` | **852,022,274 B ≈ 852 MB** | **summarised neuron→neuron×neuropil pairs — the practical one** |
| `per_neuron_neuropil_count_post_783.feather` | 233,843,050 B ≈ 234 MB | post-synapse counts per neuropil |
| `per_neuron_neuropil_count_pre_783.feather` | 16,853,770 B ≈ 17 MB | pre-synapse counts per neuropil |
| `proofread_root_ids_783.npy` | 1,114,168 B ≈ 1.1 MB | array of proofread root IDs |

Direct download URLs (verified pattern):
`https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1`
`https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1`
(the API `links.self` form is `https://zenodo.org/api/records/10676866/files/<key>/content`).
Format is **Arrow Feather** → `pandas.read_feather(path)` or `pyarrow.feather.read_table()` with chunked
`.slice()` for the 9.5 GB file. Packages: `pandas`, `pyarrow`.

**Annotation tables** — GitHub repo `https://github.com/flyconnectome/flywire_annotations` (66★, license field
empty in the GitHub API → `UNCONFIRMED`, but data is CC-BY per the FlyWire principles). Tarball of the whole repo
is the easy download. Key files:
- `supplemental_files/Supplemental_file1_neuron_annotations.tsv` — flow, superclass, cell class, nerve, lineage, side, morphology group, neurotransmitter, VFB IDs (this is the cell-type table you actually need)
- `supplemental_files/Supplemental_file2_non_neuron_annotations.tsv`, `_file3_hemilineages_clustering.csv`, `_file4_summary_with_ngl_links.csv`, `_file5_hemibrain_meta.csv`
- Skeletons + NBLAST scores are **not** in the repo: `https://doi.org/10.5281/zenodo.10877326`
- Changelog tags: annotations have been re-released (v2.1.0 = as-published; v3.0.0/v3.1.0 = Berg et al. 2025/26 cross-validation with MaleCNS). **Pick the tag matching the paper you are reproducing.**

### 1.2 Preprocessed / simplified adjacency exports — **much smaller, the ones to actually use**

- **`eonsystemspbc/fly-brain` repo data files** (770★, GPL-2.0-or-later, pushed 2026-08-29) — verified via GitHub API:
  - `data/2025_Connectivity_783.parquet` — **100,804,642 B ≈ 101 MB** — `https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/2025_Connectivity_783.parquet`
  - `data/2025_Completeness_783.csv` — ~3.2 MB (README figure; API check rate-limited, not re-verified) — neuron IDs + proofreading status, ~138k rows
  - `data/archive/2023_Connectivity_630.parquet` + `2023_Completeness_630.csv` — legacy v630 equivalents
  - Columns are essentially pre/post index + weight; loads straight into `pandas`/`torch.sparse`.
  - `https://github.com/eonsystemspbc/fly-brain`
- **mapped mesh/skeleton per-neuron downloads** — Codex per-cell SWC + synapse JSON, or `fafbseg-py`/`navis` (no bulk meshes; EM volume is far too large to download, `cloudvolume` subvolume reads only).
- **Codex static CSV exports** (gzipped): list products at `https://codex.flywire.ai/api/download?dataset=fafb`, fetch via `https://codex.flywire.ai/api/download_resource?data_product=<name>&dataset=fafb&api_token=<token>`. **Requires Google sign-in + a token from your account page.** Worked example: `https://codex.flywire.ai/programmatic_access_notebook`. Product names seen in the docs: `consolidated_cell_types`, `connections_princeton`.

### 1.3 CAVE / Codex programmatic access (live FlyWire)

- Package: **`caveclient`** (MIT, `https://github.com/CAVEconnectome/CAVEclient`, 39★, pushed 2026-09-05). `pip install caveclient` (extra: `caveclient[full]`).
- **Datastack name confirmed**: `flywire_fafb_production` (source: natverse fafbseg reference, `https://natverse.org/fafbseg/reference/flywire_cave_client.html`).
- Client code pattern (datastack name verified, exact server URL `UNCONFIRMED`):
  ```python
  from caveclient import CAVEclient
  client = CAVEclient('flywire_fafb_production')
  client.materialize.get_tables()          # -> 'synapses_v3', 'neuron_information_v2',
                                           #    'hierarchical_neuron_annotations', 'proofread_neurons', ...
  client.materialize.get_table_metadata('nuclei_v1')
  client.info.get_datastack_info().synapse_table   # -> default synapse table
  ```
- FlyWire-specific CAVE deployment seen live: `https://prod.flywire-daf.com/` (materialization, annotation, PyChunkedGraph, meshing endpoints) — reachable, no auth needed for the landing page.
- Docs: `https://caveclient.readthedocs.io/en/latest/guide/materialization.html`
- Alternative higher-level client: **`fafbseg`** (pip `fafbseg`, repo `https://github.com/navis-org/fafbseg-py` — note: NOT `flyconnectome/fafbseg-py`; that path 404s). Docs `https://fafbseg-py.readthedocs.io/`. Explicit Windows warning in the install docs; fine inside WSL2.
- Skeleton/morphology wrapper: **`navis`** (GPL-3.0, 129★) + `navis-flybrains`.

### 1.4 Janelia MaleCNS (male whole CNS, v1.0) — **easiest real download of them all**

- Public Google Cloud bucket, **no account, no token** (verified by HTTP HEAD):
  base `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
  | File | Verified size |
  |---|---|
  | `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | **508,025,642 B ≈ 508 MB** |
  | `body-neurotransmitters-male-cns-v1.0.feather` | **43,282,834 B ≈ 43 MB** |
  | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | **14,483,314 B ≈ 14 MB** |
  - Total ≈ **565 MB** — the entire wiring diagram plus cell types plus E/I sign, unblocked.
- License **CC-BY-4.0**. 166,700 neurons / 6,242,118 connections (Codex MCNS v1.0 tile); the `traced-only` variant is the double-size-filtered one, hence 508 MB.
- Website `https://male-cns.janelia.org/` (download page `https://male-cns.janelia.org/download/`); release notes `https://male-cns.janelia.org/release/`.
- Access also via **`neuprint-python`** (BSD-3-Clause, `https://github.com/connectome-neuprint/neuprint-python`, `pip install neuprint-python`):
  `Client("https://neuprint.janelia.org", dataset='male-cns:v1.0', token=token)` then `fetch_neurons`, `fetch_adjacencies`. R equivalent `neuprintr` / `malecns`. Token requires a free account.
- Working example script that downloads exactly those three files and asserts the LC4/LPLC2→DNp01 looming circuit is present: `https://github.com/Jhongdlp/FlyBrain/blob/main/fly/paso0.py`

### 1.5 Hemibrain (Janelia FlyEM, half brain)

- Paper: Scheffer et al. 2020. ~25,000 neurons, ~20M synapses, v1.2.1.
- Bulk CSV dump (verify-before-use; from `connectome-neuprint/neuPrint` README):
  `https://storage.cloud.google.com/hemibrain-release/neuprint/hemibrain_v1.0.1_neo4j_inputs.zip` or `gs://hemibrain-release/neuprint/hemibrain_v1.0.1_neo4j_inputs.zip` — **CSV, only the v1.0.1 variant confirmed in docs; v1.2.1 bulk URL `UNCONFIRMED`**.
- Live query: **`neuprint-python`** → `Client("https://neuprint.janelia.org", dataset='hemibrain:v1.2.1', token=...)`. Web UI `https://neuprint.janelia.org/`.
- Project page `https://www.janelia.org/project-team/flyem/hemibrain`.

### 1.6 BANC (female brain **and** nerve cord) + other recent releases

- BANC: ~**188,000 neurons / 199M predicted synapses**, 4 nm in-plane, unites brain + SEG + cervical connective + full VNC. Site `https://codex.flywire.ai/api/download?dataset=banc` (Codex) and `https://male-cns.janelia.org`-style project hosting at `https://github.com/htem/BANC-project` (10★, license field empty → `UNCONFIRMED`).
- Repo layout: committed metadata snapshot `data/meta/banc_888_meta_<YYYYMMDD>.parquet` (**63 MB**, per README); larger products (per-synapse tables, influence parquets, meshes) pulled on demand from the **Harvard Dataverse deposit / GCS bucket at runtime** — the exact deposit DOI is referenced only inside `data/private/` which is gitignored (so the URL is `UNCONFIRMED`; call `R/startup/` helpers to resolve it). R client `bancr` (`https://natverse.org/bancr/`), Zenodo record `https://zenodo.org/records/20350648`. Paper: Bates, Phelps, Kim, Yang et al., *Nature* 2026, `https://www.nature.com/articles/s41586-026-10735-w`.
- Other Codex-hosted datasets (all with per-dataset download pages, sizes `UNCONFIRMED`): FAFB v783 (139,255 neurons), BANC v888 (158,262), **MANC v1.2.1** male VNC (23,665), **MAOL v1.1** male right optic lobe (52,445), **MCNS v1.0** male brain+VNC (166,700).
- R-side entry point for the whole family: `coconatfly` / `coconat` (`https://github.com/natverse/coconat`, 2★, GPL-3.0) and `cocoa` (`https://github.com/flyconnectome/cocoa`, 10★, GPL-3.0).

---

## 2. Simulation / modelling frameworks

### 2.1 FlyGym / NeuroMechFly v2 (EPFL) — **actively maintained, pip-installable, Apache-2.0**

- Package: **`flygym`** (PyPI 2.1.0, wheel 3.2 MB, sdist 3.1 MB, **Apache-2.0**, repo `https://github.com/NeLy-EPFL/flygym`, 336★, pushed 2026-08-24).
- Install: `pip install flygym` / `flygym[examples]` / `flygym[rl]` / `flygym[warp]` for the GPU (MuJoCo-Warp) backend.
- What it is: a biomechanical **digital twin** — micro-CT-derived body, 1,442-ommatidia compound eye, antenna/maxillary-palp odour sensing, leg adhesion, mechanosensory feedback, and a brain→VNC hierarchical control interface. It simulates the **body and sensors**, not the connectome.
- **Meshes are lazily downloaded from S3 on first use and cached**; the packaged wheel itself is only ~3 MB (`https://neuromechfly.org/api_reference/flygym/utils/assets_lazy_loading/`). Total asset footprint `UNCONFIRMED` (README/docs do not state a figure; issue #280 tracks moving meshes to S3).
- **2.x is a March-2026 rewrite, not backward compatible**; the old 1.x API lives at `https://github.com/NeLy-EPFL/flygym-gymnasium` (`https://gymnasium.neuromechfly.org/`). Docs `https://neuromechfly.org/`; Docker image `nelyepfl/flygym`; headless rendering needs `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`. Paper: *Nature Methods* 2024, doi:10.1038/s41592-024-02497-y. `warp` extra needs CUDA → not usable on the 7900 XTX (`UNCONFIRMED` otherwise).

### 2.2 FlyBrainLab + dependencies — **effectively dormant, legacy Python only**

- Repo `https://github.com/FlyBrainLab/FlyBrainLab` (93★, BSD-3-Clause, last push **2025-09-29**). Components `Neuroballad` (3★, BSD-3), `FBLClient` (3★, BSD-3) — all last touched 2025-09-29, i.e. maintained only by a bulk commit, no recent development.
- **Hard Python 3.9 pin**: `conda create -n flybrainlab python=3.9` then `pip install git+.../Neuroballad.git nxt_gem==2.0.1 git+.../nxcontrol flybrainlab[full] neuromynerva`. This will not resolve on a modern Python; you need a conda env pinned to 3.9.
- **Full install requires Ubuntu 16.04+, a CUDA GPU, ≥30 GB disk and recommends ≥32 GB RAM** (OrientDB Java heap + disk cache) → over this box's free-disk budget at the recommended setting, and CUDA-only. Docker route: `docker pull fruitflybrain/fbl:latest`, `docker run --gpus all -p 9999:8888`. Docker image and AMI `ami-02218ae5a3d1fd06d` are the maintained artifacts.
- Dependency state: `FlyBrainLab/GFX` and `FlyBrainLab/GFXFormats` **return 404 from the GitHub API → renamed/moved or deleted; GFX is not installable as documented** (`UNCONFIRMED` where it went). `Neurokernel` lives at `https://github.com/neurokernel/neurokernel` (578★, non-standard license `NOASSERTION`, last push 2025-09-29) — same dormancy; the FlyBrainLab README's GPU Neurokernel execution is CUDA-only.
- **Verdict: not recommended on this hardware.** 30 GB + CUDA + Python 3.9 for a circuit-visualisation tool.

### 2.3 Whole-brain LIF simulators — **the actual "run the fly brain" projects**

| Project | Repo / license | Scale | Notes |
|---|---|---|---|
| **Shiu et al. original** | `https://github.com/philshiu/Drosophila_brain_model` (286★, **MIT**, last push 2024-09-14) | whole brain, LIF | The canonical paper code. Paper: *Nature* 2024, doi:10.1038/s41586-024-07763-9. Brian2. Inactive since 2024. |
| **`eonsystemspbc/fly-brain`** | `https://github.com/eonsystemspbc/fly-brain` (770★, **GPL-2.0-or-later**, pushed 2026-08-29) | ~138k neurons, ~5M synapses | Ships the **101 MB parquet + 3.2 MB CSV** in-repo. Six backends benchmarked: Brian2 (CPU), Brian2CUDA, PyTorch, NEST GPU, GeNN/PyGeNN, Brian2GeNN. README documents a working **WSL2 + Ubuntu 22.04** setup (`scripts/setup_WSL_CUDA.sh`) — but every accelerated backend is CUDA, so **only the Brian2 CPU path runs here**. Derived parquet ≈ 288 MB COO / 289 MB CSR. |
| **`annel0/flybrain`** | `https://github.com/annel0/flybrain` (2★, **MIT** code; data not redistributed) | MaleCNS v1.0, 166,700 neurons | Hand-written **Triton** kernels, CSR, fp16; claims **2.4× realtime for one full CNS** on a single RTX 3060 12 GB, compiled weights ~1.2 GB. Triton is CUDA-only → **will not run on ROCm/WSL2**. Contains an honest negative-results notebook, incl. a rewiring control showing APL goes completely silent and the mushroom-body loop disappears without real wiring. |
| **`suifei/flywire-fly-lab`** | `https://github.com/suifei/flywire-fly-lab` (**GPL-3.0-or-later**; `results/` data is **CC-BY-NC-4.0, non-commercial**) | 138,639 neurons / 15,091,983 weighted edges, dt = 0.1 ms | **Explicitly done on a 16 GB M1 Pro: ~2.5 s of wall time per 1 s of brain, peak 3–4 GB RAM.** This is the size baseline that matters for your 31 GB box. Includes a 17,628-target virtual-knockout screen, a BANC/MANC VNC model, a flyvis→whole-brain vision pipeline, and `language/` ("fly speaks Chinese", 12-word decoder + BCI stress test). Browser sub-circuit demo = 4,599 neurons / 338,837 edges at 13× realtime. |

### 2.4 Mushroom body / olfactory circuit models

- **`annkennedy/mushroomBody`** — `https://github.com/annkennedy/mushroomBody`. ORNs → PNs → GABAergic antennal-lobe local neurons → Kenyon cells → (APL-style feedback) dynamic MB model. Exact mechanics are **MATLAB-based** (`.m` files per the companion write-up) → MATLAB/Octave, not Python (`UNCONFIRMED` whether a Python port exists).
- **`sachinksalim/drosophilia_mushroom_body`** — `https://github.com/sachinksalim/drosophilia_mushroom_body` (~2000 KCs, ~6 random of ~54 PN channels, APL local vs global inhibition, LIF-2D KC model, softmax separability readout). Report: `https://sachinksalim.github.io/files/reports/um/Fruitfly_Mushroombody.pdf`. Small, readable, and directly the "sparse expansion coding = hidden layer" experiment.
- **Locust olfactory / MB in NEURON** — `https://github.com/ModelDBRepository/262670` (GGN + MB circuit, NEURON 7.4). Background: `https://pmc.ncbi.nlm.nih.gov/articles/PMC4070380/` (large-scale locust AL model). Classic sparse-coding reference implementation, dated toolchain.
- **`brain2`/NEST recipes** — the fly-brain repo above is the best-maintained Brian2/NEST fly example; there is no separate well-maintained "mushroom body in NEST" package (`UNCONFIRMED`).

---

## 3. Connectome-constrained / connectome-derived networks for computation

### 3.1 Flyvis (Turaga Lab) — **best-maintained connectome-constrained *learnable* network**

- `https://github.com/TuragaLab/flyvis` (165★, **MIT**, pushed 2026-08-18); **`pip install flyvis`** (PyPI 1.2.0, wheel 385.6 kB, sdist 30.1 MB, uploaded 2026-08-06). PyTorch.
- The official implementation of Lappalainen et al., *Nature* 2024, doi:10.1038/s41586-024-07939-3: connectivity of 64 cell types in the fly motion pathways wired as a differentiable network, with unknown biophysical parameters fit by gradient descent.
- This is the closest thing to **"connectome as a network architecture"**: the wiring is a fixed sparse structure, only parameters are learned. Pretrained model ensembles ship with it (checkpoint download size `UNCONFIRMED`). 7 tutorials incl. "Train the Network on the Optic Flow Task"; docs `https://turagalab.github.io/flyvis/`.
- Scope: optic lobe / motion vision only (not whole brain, not language).

### 3.2 "Connectome as a neural-network layer" / language-like prior art

- **FLM — Fly Language Model** — `https://github.com/nftechie/flm` (**MIT** code; weights/data retain upstream terms). MaleCNS v1.0 as a **frozen recurrent graph**: 166,700 nodes, 25,582,938 directed connections, recurrence `x = tanh(W @ (0.6x + 0.4·input))` with incoming-normalised anatomical contact counts; a **278,528-parameter adapter** reads graph state and biases next-token logits of **Liquid AI LFM2.5-1.2B-Instruct**. Only the adapter trains. Needs Python 3.12, ≥10 GB disk, ≥16 GB RAM; downloads pinned to upstream revisions with SHA-256 checks. **Honest caveat in its own README: the matched direct-input control performed slightly *better*, so it does not establish an advantage from fly anatomy.** Paper: `https://artificialscientific.com/papers/flies-are-all-you-need`.
- **`suifei/flywire-fly-lab` `language/`** — 12-word decoder + brain-computer-interface stress test ("果蝇说中文"), over a full 138k-neuron Brian2/PyTorch LIF model. **CC-BY-NC-4.0 data, GPL-3.0 code.**
- **`Jhongdlp/FlyBrain`** — `https://github.com/Jhongdlp/FlyBrain` (**MIT**). MaleCNS v1.0 in sparse CSR/CSC, vectorised LIF, a 1,442-ommatidia retina, `LC4`+`LPLC2`→`DNp01` looming escape, 381 VNC leg motor neurons, cVA→`DA1`→Kenyon cells→`MBON` aggression, PAM/PPL1 dopamine depression of KC→MBON. Rust engine (+ WASM/PyO3), bit-for-bit deterministic, >1M steps/s headless. Downloads the same **540 MB Janelia bucket** shown in §1.4.
- **Adjacent prior art (same idea, other people's brains/circuits)**: `nftechie/doomfly` (fly connectome → live Doom arena), `seanphan/flyt3` (MaleCNS tic-tac-toe on k3s), `flybrain.online`, `mrfly.dev`, `malecns.io` (MaleCNS in Rust, 165,122 neurons, "browsing the open internet"), `zenabruh/flybraincm` (139,255 neurons / 2.7M connections browser sim). "The Digital Sphinx" (Brunton & Tuthill 2026, cited by `annel0/flybrain`) attached a *worm* connectome to a fly body with a trained interface — cited there as evidence that behavioural realism alone proves nothing.
- **Other academic lines**: arXiv 2602.17997 "Whole-Brain Connectomic Graph Model Enables Whole-Body…" (Fly-connectomic Graph Model) — abstract-level only, `UNCONFIRMED` code availability; OpenReview `https://openreview.net/forum?id=wCBNxp1qWe` (online fitting of connectome-constrained whole-brain models, 2025) — `UNCONFIRMED` code.

---

## 4. Verdict — the single most practical combination

**Download the Janelia MaleCNS v1.0 flat-connectome (three `.feather` files, ~565 MB total, no account, no token, CC-BY-4.0, verified reachable) and run it with `suifei/flywire-fly-lab`'s whole-brain LIF recipe (or the `eonsystemspbc/fly-brain` Brian2 CPU path, whose 101 MB pre-baked v783 parquet you can also just grab from GitHub).** Roughly 600 MB–1 GB of downloads, all of it plain HTTP/Feather that `pandas.read_feather` opens directly, and — critically for a 31 GB / AMD machine — a **documented CPU-only data point**: `flywire-fly-lab` ran 138,639 neurons and 15.1M weighted connections at ~2.5 s of compute per simulated second in **3–4 GB peak RAM on a 16 GB laptop**, so your box has ~8× headroom and needs no GPU whatsoever. The same MaleCNS files need no authentication, whereas FlyWire's canonical release (Zenodo 10676866) costs you 852 MB minimum and the full-fidelity version 9.5 GB, and Codex/CAVE downloads plus the hemibrain bulk dump both require sign-in flows. If you want the *learnable* connectome-constrained network rather than a pure simulation, add `pip install flyvis` (MIT, 386 kB wheel) as a second stack — it is small, actively maintained, and PyTorch-based, so it is the one GPU-capable component that could plausibly use the 7900 XTX via ROCm. **What I would not do on this hardware: FlyBrainLab (30 GB + CUDA + Python 3.9) or anything CUDA-only (Brian2CUDA, NEST GPU, GeNN, Triton, MuJoCo-Warp).**

---

### Verification status

**Verified live** (HTTP `Content-Length` / GitHub REST API / Zenodo REST API / fetched page text):
Zenodo 10676866 file list and all five sizes; CC-BY-4.0 license; MaleCNS bucket URLs and all three sizes; `flywire_fafb_production` datastack name; `eonsystemspbc/fly-brain` parquet size (100,804,642 B); all repo star counts/push dates/licenses/PyPI versions in the tables above; flygym 2.1.0 and flyvis 1.2.0 metadata; Codex download-endpoint and token flow; BANC 188k/199M and 63 MB metadata parquet (from its README).

**Flagged `UNCONFIRMED`**: `data/2025_Completeness_783.csv` size (GitHub API rate-limited after the first call; 3.2 MB is the repo README's own figure); hemibrain v1.2.1 bulk-download URL; BANC Dataverse/GCS deposit URL (gitignored); FlyGym total mesh-asset size; flyvis pretrained checkpoint size; FlyBrainLab `GFX`/`GFXFormats` current location (both 404); `flywire_annotations` and `BANC-project` licenses (empty SPDX field); whether ROCm-on-WSL2 works for `gfx1100` on this box; arXiv 2602.17997 and the OpenReview 2025 item's code availability.
