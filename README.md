# PSI Field Lab

### Synthetic Calibration and Prospective Field Validation Protocol

**Paranormal 3:33 Research Lab** · Methodology paper v1.0 · September 2026 · Author: **Alberto Cardenas**

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Data](https://img.shields.io/badge/data-100%25%20synthetic-orange.svg)
![Seed](https://img.shields.io/badge/seed-333-purple.svg)
![DOI](https://img.shields.io/badge/DOI-pending-yellow.svg)

> **Core boundary statement.** This repository contains a *documentation, calibration, and field-validation methodology*. It is **not** designed to prove or disprove paranormal causation. A high score means **unusual under the recorded conditions**, never **paranormal**.

> **Synthetic-data notice.** Every record in this repository was generated computationally (`generate_psi_data_v2.py`, seed `333`). No participant wore a headset or heart-rate monitor, no camera or recorder captured a location, and no environmental probe was read in the field. All results shown here are **synthetic calibration results** and must never be presented as field evidence.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [What we are looking for](#2-what-we-are-looking-for)
3. [Original contribution](#3-original-contribution)
4. [Repository contents](#4-repository-contents)
5. [Quick start: reproducing the lab](#5-quick-start-reproducing-the-lab)
6. [Configuration reference (`.env`)](#6-configuration-reference-env)
7. [Stage I: the synthetic calibration dataset](#7-stage-i-the-synthetic-calibration-dataset)
8. [The analytical pipeline, step by step](#8-the-analytical-pipeline-step-by-step)
9. [Reference run results (synthetic)](#9-reference-run-results-synthetic)
10. [Run package and audit trail](#10-run-package-and-audit-trail)
11. [Stage II: prospective real-data field protocol](#11-stage-ii-prospective-real-data-field-protocol)
12. [Preregistration, governance, and ethics](#12-preregistration-governance-and-ethics)
13. [Limitations and transition criteria](#13-limitations-and-transition-criteria)
14. [Security and secrets](#14-security-and-secrets)
15. [Archiving on Zenodo, OSF Preprints, and ResearchGate](#15-archiving-on-zenodo-osf-preprints-and-researchgate)
16. [How to cite](#16-how-to-cite)
17. [License](#17-license)
18. [References](#18-references)
19. [Contact](#19-contact)

---

## 1. What this project is

Paranormal and parapsychological field research is often pulled toward two opposite errors. One begins with belief and reads every unusual observation as evidence. The other begins with rejection and treats every unusual observation as error. The **PSI Field Lab** takes a third position: *the unknown deserves disciplined documentation before interpretation.*

An event can be unusual, unresolved, or worth following up without becoming evidence of spirits, survival of consciousness, telepathy, or any other extraordinary cause. The methodology keeps four levels apart that are usually mixed together:

| Level | Meaning | Example |
|---|---|---|
| **Observation** | What a record states | A timestamp, a heart-rate delta, an audio transient, a motion score |
| **Interpretation** | A provisional human reading of that record | "The participant appears startled" |
| **Explanation** | A conventional or unresolved account that may fit | "Substation hum", "radio bleed", "unresolved" |
| **Claim** | A stronger statement about cause | *Not permitted in this framework* |

The PSI Field Lab works only with **observations, interpretations, and review priorities**. No rule, statistic, machine-learning model, or language-model reviewer in this repository is authorized to make a paranormal causal claim.

The framework has **two stages**:

```
 STAGE I  (this repository, runnable today)          STAGE II  (prospective protocol)
 ┌───────────────────────────────────────┐           ┌──────────────────────────────────────────┐
 │ Synthetic calibration bench           │           │ Real-data field validation               │
 │  • fixed seed 333, known answers      │  ──────►  │  • Muse 2, Polar RR, body cams, audio,   │
 │  • tests loading, gates, scoring,     │  bridge:  │    environmental instruments             │
 │    ML, site repeatability, LLM review,│  same     │  • matched target vs control sessions    │
 │    sensitivity, audit packaging       │  10-s     │  • blinding, preregistration, repetition │
 │  • NO real sensor data                │  schema   │  • participant = experimental unit       │
 └───────────────────────────────────────┘           └──────────────────────────────────────────┘
```

The methodological contribution is a **traceable bridge** from synthetic software calibration to preregistered field acquisition. Real field data will be transformed into the same ten-second analytical schema, so the pipeline validated in Stage I can run unchanged in Stage II, with its rules frozen before any real data is seen.

---

## 2. What we are looking for

### 2.1 The Stage I question (software calibration)

Can the analytical pipeline, under conditions whose answers are known in advance:

1. **elevate** designed multimodal candidates (convergent physiological, EEG-style, and audio deviations with low motion and no conventional source),
2. **demote** known artifacts: movement bursts, degraded signal quality, Spirit Box radio bleed, documented electrical/environmental sources, and startle responses where the sound *precedes* the physiology,
3. **distinguish** genuine cross-participant repetition at a site from repeated activity in *one* participant, and from "hot" sites that are really infrastructure clusters,
4. **expose** poor model performance honestly (for example, a weak unsupervised detector),
5. **preserve** every result in a run-specific, hashed, auditable package?

The notebook frames this as six research questions:

1. Which events show convergent deviations across independent modalities?
2. Do unusual events repeat across *different* participants at the same site?
3. Can an unsupervised model identify the same candidates without paranormal labels?
4. How much do environmental confounders and independent audio review change the ranking?
5. Are the final rankings robust to reasonable changes in scoring weights?
6. Does the LLM reviewer agree with the quantitative triage, and does it stay inside the evidence?

### 2.2 The Stage II question (empirical, bounded)

> Under prospectively fixed rules, do predefined multimodal candidate patterns occur more consistently in **target-site** sessions than in **matched control** sessions, and do they **replicate across independent participants and collection waves**?

This does not assume a paranormal cause. It asks whether an observable, quality-controlled pattern repeats after conventional explanations, movement, device failure, and environmental context have been considered. Even a fully supported result is described as *a replicated unexplained multimodal pattern under study conditions*, never as a paranormal cause.

---

## 3. Original contribution

The paper contributes a specific framework, not a claim of discovery. Its original contribution is the combination of:

- a **two-stage sequence** from synthetic software calibration to prospective real-data field validation;
- explicit **separation of observation, interpretation, explanation, and causal claim**;
- **multimodal sensor alignment** using ten-second analytical windows and immutable raw archives;
- **gate-first scoring**, where quality, motion, and conventional explanations are eligibility conditions rather than evidence points;
- **participant-level site repeatability** rather than raw event counts;
- **constrained language-model review** as an audit aid, not an oracle;
- **run-level packaging** with manifests, hashes, configuration, prompts, model identity, and visible limitations;
- a **governance model** for preregistration, consent, participant wellbeing, and controlled data release.

---

## 4. Repository contents

```
Paranormal-333-PSI-Field-Lab/
├── README.md                                  ← this document
├── PSI_Field_Lab_Methodology_Paper_v1.0.pdf   ← the methodology paper (version of record)
├── PSI_Lab_v3_LLM.ipynb                       ← analysis notebook (with outputs of the reference run)
├── generate_psi_data_v2.py                    ← synthetic dataset generator (seed 333)
├── requirements.txt                           ← Python dependencies
├── CITATION.cff                               ← machine-readable citation (GitHub / Zenodo)
├── LICENSE                                    ← CC BY-NC-ND 4.0 legal code
├── .gitignore                                 ← excludes paranormal/.env and run outputs
├── .gitattributes                             ← keeps CSVs byte-identical across platforms
└── paranormal/                                ← data folder (the notebook's PSI_DATA_DIR)
    ├── .env.example                           ← configuration template (copy to .env)
    ├── README.md                              ← dataset design notes
    ├── sessions.csv                           ← 162 rows × 33 cols
    ├── sensor_windows.csv                     ← 28,122 rows × 29 cols
    ├── events.csv                             ← 220 rows × 25 cols
    ├── audio_reviews.csv                      ← 160 rows × 17 cols
    └── ground_truth.csv                       ← 220 rows (evaluation only)
```

`paranormal/.env` (your real credentials) exists only on your machine and is **git-ignored**.

---

## 5. Quick start: reproducing the lab

Reproduction has two independent levels:

- **Level A, dataset reproducibility (exact).** Regenerating the synthetic data with seed `333` reproduces the published CSVs byte-for-byte (after line-ending normalization). Needs only `numpy` + `pandas`.
- **Level B, analysis reproducibility.** Rerunning the notebook reproduces every rule-based, statistical, and ML result exactly (all random steps are seeded with `333`). LLM outputs depend on the model and endpoint; the notebook records the model identity and keeps the raw responses so they can be audited. An offline **mock mode** exercises the full pipeline without any model.

### 5.1 Get the code

```bash
git clone https://github.com/betoalien/Paranormal-333-PSI-Field-Lab.git
cd Paranormal-333-PSI-Field-Lab
```

### 5.2 Create an environment and install dependencies

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.3 (Level A) Regenerate the synthetic dataset and verify it

```bash
python generate_psi_data_v2.py --out paranormal_v2 --seed 333
```

Expected console output (abridged):

```
seed=333 -> .../paranormal_v2
sessions (162, 33) | windows (28122, 29) | events (220, 25) | audio_reviews (160, 17)
```

Verify the published files against these SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `paranormal/sessions.csv` | `556e2292258cfbd8713eb833c70b2dc777bbaed41ca9229e07a2bd9b271dd5a1` |
| `paranormal/sensor_windows.csv` | `ea519568fec7172e89a7fd44993f3629cb1e56bffe2cee436be82d216fcf75ad` |
| `paranormal/events.csv` | `9c6c0317e211124b83103a6de807893b5c5192c7df1ce7f62836e830241aca19` |
| `paranormal/audio_reviews.csv` | `1f184961ae8736054488666e1e879d85bc682092948b17617be230b675e7bef8` |
| `paranormal/ground_truth.csv` | `8fbb093f8de75b5a2caf8c18b9674505bf7ef0a392492155bdef0a8460d149d7` |
| `generate_psi_data_v2.py` | `e497599c63b77994322fc0d794ad42d21df3f6d1bd71dd792afbd355b16bc138` |
| `PSI_Field_Lab_Methodology_Paper_v1.0.pdf` | `1fc7ccb6965c86d11b4b2c278a6e1b5befdbedf98076ea44feb95da103ac3b1e` |

```bash
# macOS / Linux / Git Bash
sha256sum paranormal/*.csv

# Compare a regenerated file with the published one, ignoring line endings
# (on Windows, pandas writes CRLF; the published files use LF)
diff <(tr -d '\r' < paranormal_v2/events.csv) paranormal/events.csv && echo IDENTICAL
```

```powershell
# Windows PowerShell
Get-FileHash paranormal\*.csv -Algorithm SHA256
```

> Verified: regenerating on Python 3.12.7 / numpy 2.4.2 / pandas 3.0.0 (Windows 11) produced content identical to all five published CSVs. A **different seed** gives a new realization of the same experimental design, which is useful for measuring pipeline variance across datasets.

### 5.4 Configure the environment file

```bash
cp paranormal/.env.example paranormal/.env
```

Then edit `paranormal/.env`. The template defaults are chosen so the lab runs **completely offline**:

- `USE_POSTGRES=false`: read the CSV files in `PSI_DATA_DIR`
- `PSI_LLM_MODE=auto`: use the LLM if configured, otherwise fall back to the labelled mock

To use a real language model, set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and (for a hosted endpoint) `OLLAMA_API_KEY`. See [Section 6](#6-configuration-reference-env) for every variable.

### 5.5 (Level B) Run the analysis notebook

The notebook finds its configuration file automatically, so it runs **unchanged on any machine**. It loads the first `.env` it finds in this order:

1. the path in the `PSI_ENV_FILE` environment variable, if set;
2. `./paranormal/.env`, when the working directory is the repository root;
3. `../paranormal/.env`, when the notebook is opened from a subfolder;
4. `/work/paranormal/.env`, the maintainers' internal Deepnote workspace.

If none exists, it stops with a message listing every path it searched. When `PSI_DATA_DIR` is not set, the data folder defaults to the folder containing the `.env` (`paranormal/`).

#### Option 1: Local machine (Jupyter), recommended

```bash
cp paranormal/.env.example paranormal/.env   # if not done yet
jupyter lab PSI_Lab_v3_LLM.ipynb             # Run → Run All Cells
```

The template's defaults (`PSI_DATA_DIR=./paranormal`, `PSI_OUTPUT_DIR=./psi_lab_v3_outputs`, `PSI_DOWNLOAD_DIR=./psi_lab_v3_downloads`) are relative to the repository root, and both output folders are git-ignored. On Windows, write absolute paths in `.env` with forward slashes (`C:/data/paranormal`), because backslashes are treated as escape characters.

To keep the `.env` somewhere else, point to it before starting Jupyter:

```bash
export PSI_ENV_FILE=/path/to/my.env          # PowerShell: $env:PSI_ENV_FILE="C:/path/to/my.env"
```

#### Option 2: Google Colab

Upload the notebook and the `paranormal/` folder to `/content/` (the default working directory), then run all cells; `./paranormal/.env` is found automatically. Alternatively set `os.environ["PSI_ENV_FILE"]` in a first cell.

#### Option 3: Internal Deepnote workspace (maintainers only)

The reference run was executed in the Paranormal 3:33 Research Lab Deepnote project, where the files live at `/work/paranormal/` and the private `.env` sets `PSI_DATA_DIR=/work/paranormal`, `PSI_OUTPUT_DIR=/tmp/psi_lab_v3_outputs`, and `PSI_DOWNLOAD_DIR=/work/psi_lab_v3_downloads`. External users do not need this layout.

### 5.6 LLM modes

| `PSI_LLM_MODE` | Behavior | When to use |
|---|---|---|
| `auto` | Tries the endpoint with a controlled ping; falls back to `mock` on failure | Default for collaborators |
| `live` | Requires a working endpoint; **hard error** otherwise, so mock output can never be mistaken for model output | Reference / publication runs |
| `mock` | Never calls a model. A transparent, deterministic heuristic returns schema-valid JSON so every downstream cell can be exercised. Every mock response is labelled `MOCK OUTPUT — deterministic heuristic, not a language model` | Offline reproduction, CI, teaching |

Output files carry the mode in their names (`llm_review_summary_live.csv` vs `llm_review_summary_mock.csv`), so the two cannot be confused.

### 5.7 Optional: PostgreSQL / Neon data source

With `USE_POSTGRES=true` and a valid `DATABASE_URL`, the notebook reads the five tables from `POSTGRES_SCHEMA` (default `psi_lab`) instead of CSV files. It still exports portable CSV snapshots (`source_tables_csv/`) so collaborators never need database access. Schema and table names pass through an identifier whitelist (`^[A-Za-z_][A-Za-z0-9_]*$`) before being used in SQL. Setting `WRITE_ANALYSIS_RESULTS_TO_POSTGRES=true` also persists `feature_scores`, `llm_reviews`, and `pipeline_runs`.

---

## 6. Configuration reference (`.env`)

All configuration is read from `paranormal/.env`. **No credential is ever hard-coded in the notebook.**

| Variable | Default | Used for |
|---|---|---|
| `PSI_ENV_FILE` | *(unset)* | Optional explicit path to the `.env` file (set as an OS environment variable, not inside `.env`) |
| `ENVIRONMENT` | `local` | Label printed at start-up |
| `PSI_DATA_DIR` | folder of the `.env` | Folder with the five CSV files |
| `PSI_OUTPUT_DIR` | `psi_lab_v3_outputs` | Root of per-run output folders |
| `PSI_DOWNLOAD_DIR` | `psi_lab_v3_downloads` | Where the run ZIP is copied |
| `SEED` | `333` | Seed for Isolation Forest, bootstrap, review-set sampling, and weight sensitivity |
| `PSI_LLM_MODE` | `auto` | `auto` / `live` / `mock` (see 5.6) |
| `OLLAMA_BASE_URL` | none | Ollama-compatible endpoint |
| `OLLAMA_MODEL` | none | Model name (reference run: `gemma4:latest`) |
| `OLLAMA_API_KEY` | none | Bearer token for hosted endpoints (**secret**) |
| `LLM_TEMPERATURE` | `0` | Sampling temperature for every LLM call |
| `LLM_REVIEW_REPEATS` / `PSI_LLM_REPEATS` | `3` | Repeated calls per event (majority vote, self-agreement) |
| `PSI_LLM_TOP_N` / `LLM_REVIEW_TOP_N` | `20` | Top-ranked events reviewed (+ the same number of random lower-ranked controls) |
| `LLM_BATCH_SIZE` | `25` (template: `10`) | Events per request in the blind batch cross-check |
| `PSI_BLIND_SALT` | `psi-333` | Salt for hashing site/participant IDs in blind records |
| `USE_POSTGRES` | `false` | Read tables from PostgreSQL instead of CSV |
| `DATABASE_URL` | none | SQLAlchemy URL (**secret**) |
| `POSTGRES_SCHEMA` | `psi_lab` | Schema holding the tables |
| `DB_WRITE_MODE` / `DB_ECHO` | `append` / `false` | SQLAlchemy behavior |
| `WRITE_ANALYSIS_RESULTS_TO_POSTGRES` | `false` | Persist derived tables to the database |
| `RESULTS_DB_WRITE_MODE` | `replace` | `if_exists` mode for derived tables |
| `TABLE_*` | table names | Override table names in PostgreSQL mode |
| `PIPELINE_NAME`, `PIPELINE_VERSION`, `PROMPT_VERSION`, `DATASET_VERSION` | see template | Written into `pipeline_run_metadata.json` |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD/SSLMODE/CHANNEL_BINDING` | none | Documentation of the connection; only `DATABASE_URL` is read by code |
| `STORE_*`, `CHECK_INVENTED_NUMBERS`, `HUMAN_REVIEW_REQUIRED`, `LLM_SAVE_RAW_RESPONSE`, `LLM_VALIDATE_JSON`, `DEMO_*` | `true` / labels | Governance flags recorded for audit. The current notebook always performs these steps. |

---

## 7. Stage I: the synthetic calibration dataset

### 7.1 Provenance

The dataset is generated by `generate_psi_data_v2.py` with random seed **333**. The generator builds a structured experimental universe (15 sites, 10 participant codes, 150 primary sessions plus 12 repeat sessions) and writes five canonical CSV files. **Events are derived from the sensor windows** (deltas computed from the windows), not the other way around, which mirrors how a real acquisition-and-transformation pipeline would work.

| File | Rows | Role |
|---|---:|---|
| `sessions.csv` | 162 | Protocol and contextual metadata: 150 trial-A sessions + 12 trial-B repeats of interrupted sessions |
| `sensor_windows.csv` | 28,122 | Ten-second windows with simulated physiological, EEG-style, motion, audio, environmental, and signal-quality features |
| `events.csv` | 220 | Timestamped candidates with deltas computed against the 60 s preceding each event |
| `audio_reviews.csv` | 160 | Simulated independent audio reviews: every event with audio plus a 20% random control sample |
| `ground_truth.csv` | 220 | Generator-only design labels. **Loaded only after all analysis; never fed to scoring or the LLM.** |

### 7.2 Session structure

Each simulated session follows the Stage II protocol: a **5-minute baseline** (`phase = before`), a **20-minute active period** (`during`), and a **5-minute recovery** (`after`), sampled in non-overlapping 10-second windows.

### 7.3 Intended sensor correspondence

Every synthetic feature family has a real counterpart that Stage II will collect:

| Feature family | Intended real source | Required transformation |
|---|---|---|
| Heart rate, RR, RMSSD, SDNN | Polar heart-rate system with beat-to-beat RR export | Preserve raw RR intervals, flag invalid beats under frozen rules, compute windowed HR/HRV |
| Relative EEG-style bands (δ, θ, α, β, γ) and accelerometry | Muse 2 EEG and accelerometer streams | Retain raw channels, document contact quality, reject motion-contaminated periods, estimate spectral power, normalize within windows |
| Motion and light-change scores | Timestamped body-camera video | Preserve original video, compute frame-level motion and luminance change, aggregate to the window |
| Audio RMS, peaks, transients | Independent timestamped audio recorder | Preserve source audio, document gain and sample rate, extract acoustic features, retain clips for blinded review |
| Environmental and contextual covariates | Environmental instruments and structured field log | Align device logs and manual observations to the common clock; record known conventional sources |

### 7.4 Data dictionary (summary)

| Table | Representative fields |
|---|---|
| **sessions** | `session_id`, `site_id`, `site_name`, `participant_id`, `trial`, `repeat_of_session_id`, date/time, planned/actual duration, `session_status`, `primary_analysis_eligible`, device statuses (`bodycam_status`, `muse2_status`, `polar_status`, `recording_sync_status`), building type/material/occupancy, electricity, running water, traffic, cellular signal, temperature, humidity, weather, rainfall, wind, ambient noise, known animals, known electrical sources, `protocol_deviation`, `investigator_notes` |
| **sensor_windows** | `session_id`, `window_id`, start/end seconds, `phase`, `investigation_zone`, `heart_rate_bpm`, `rr_mean_ms`, `rmssd_ms`, `sdnn_ms`, `muse_{delta,theta,alpha,beta,gamma}_rel` (sum to 1), `muse_accel_mean_g`, `muse_accel_peak_g`, `bodycam_motion_score`, `bodycam_light_change_score`, `audio_rms_dbfs`, `audio_peak_dbfs`, `audio_transient_count`, temperature, humidity, `environmental_noise_db`, per-device signal quality, `artifact_flag` |
| **events** | `event_id`, session/site/participant IDs, start/end seconds, phase, zone, `event_type`, `source_modalities`, `detection_method`, `observation_text`, `explanation_status`, `known_explanation`, `environmental_match`, `manual_marker`, `signal_quality_summary`, `hr_delta_bpm_vs_pre_event`, `rmssd_delta_pct_vs_pre_event`, `beta_delta_pct_vs_pre_event`, audio peak/transients, motion, noise, `review_note` |
| **audio_reviews** | `audio_review_id`, event/session/site/participant IDs, review interval, `recorder_status`, `spirit_box_status`, `audio_result`, `audible_content`, `speech_intelligibility`, `radio_bleed_possible`, `environmental_source_possible`, `reviewer_confidence`, `explanation_status`, `review_notes` |
| **ground_truth** | `event_id`, `session_id`, `generator_label`, `is_designed_anomaly`, `designed_cluster`, `audio_design`. Evaluation only. |

### 7.5 Designed labels (the "known answers")

**Designed positives** (`is_designed_anomaly = True`, 34 events):

| Label | n | Design |
|---|---:|---|
| `tp_full` | 16 | Three physiological modalities + audio, low motion, no environmental explanation |
| `tp_partial` | 12 | Two of three modalities, values close to the thresholds (hard to detect) |
| `tp_no_audio` | 6 | Physiological convergence; the independent recorder captures nothing |

**Hard negatives** (48 events, which should stay **out** of the top ranks):

| Label | n | Why it is a trap |
|---|---:|---|
| `hn_movement` | 12 | Full physiology explained by a movement burst (motion 0.30–0.50) |
| `hn_degraded_quality` | 8 | Beta/HR spike with Muse quality 0.45–0.65 and `artifact_flag` |
| `hn_spiritbox_radio` | 8 | `voice_like` audio with the Spirit Box **on**; mild physiology |
| `hn_env_confounded` | 12 | Full convergence during a documented electrical interval. 5 at SITE_01 and 4 at SITE_09 form false "infrastructure clusters" |
| `hn_startle_lag` | 8 | The audio transient occurs ~10 s **before** the physiology (a conventional startle). Only distinguishable in `sensor_windows`; in `events.csv` it looks like `tp_full` |

**Background** (`bg_*`, 126 events) and `protocol_interruption` (12 events in ineligible sessions).

**Site structure** (`designed_cluster`):

| Cluster | Design | The pipeline should… |
|---|---|---|
| `SITE_14_independent_participants` | Positives across 8 different participants | …rank it first: genuine cross-participant repeatability |
| `SITE_15_single_participant` | 3 of 4 positives in the **same** participant | …*not* treat it as repeatability |
| `SITE_01` / `SITE_09_infrastructure_cluster` | Look "hot" until context is applied | …demote them once environmental sources are gated |
| `isolated` | Single positives at sites with no repetition | …leave them unremarkable at site level |

### 7.6 What Stage I can and cannot establish

**Can test:** loading, joins, event deltas, session baselines, quality gates, ranking behavior, participant-level site repetition, sensitivity reporting, constrained LLM review, and run packaging.

**Cannot validate:** sensor accuracy, synchronization under real clock drift, electrode contact, RR artifact correction, camera occlusion, audio contamination, investigator behavior, participant expectation, site effects, or **any paranormal hypothesis**.

> **Protocol rule:** all demonstration results must be labeled *synthetic* in the notebook, paper, tables, figures, and exported summaries. No synthetic performance metric may be presented as field evidence.

---

## 8. The analytical pipeline, step by step

The notebook (`PSI_Lab_v3_LLM.ipynb`) is organized in 15 sections. Every threshold below is a fixed, visible constant, so each one can be frozen and preregistered.

### §1–2 · Environment, data location, private LLM configuration
Locates and loads `.env` (see 5.5), resolves data and output directories, assigns a **UTC run ID** (`YYYYMMDD_HHMMSS_UTC`), and creates a dedicated output folder so one run never overwrites another. Tests the LLM connection with a controlled prompt.

### §3 · Load and verify the research tables
Loads `sessions`, `sensor_windows`, `events`, `audio_reviews` (**not** `ground_truth`). Integrity checks run before anything is scored and stop the run on failure:
- foreign keys (`sensor_windows`/`events` → `sessions`; `audio_reviews` → `events`)
- unique `event_id`; at most one audio review per event
- non-negative event durations; no missing ML features
- EEG relative band powers sum to 1 (±0.001)

Audio-review coverage is reported per event type so that **a missing review is never confused with a negative review**.

### §4 · Single-session walkthrough
Session `SITE_14_P01_A` is plotted (HR, RMSSD, relative beta, audio peak, body-cam motion, Muse quality) with event intervals and phase boundaries. This is a human audit layer: a numeric anomaly can still be movement, dropout, clipping, or an obvious environmental transition.

### §5 · Window-level detection (an independent route to candidates)
`events.csv` is a field annotation, and this section finds candidates directly from the raw 10-second windows:
1. Each session's **baseline phase** gives a participant-specific mean and SD per signal.
2. Every *during* window gets a directional z-score: **HR ↑**, **RMSSD ↓**, **β ↑**.
3. Signal-specific SD floors prevent small-scale features from being suppressed: HR = 1.0 BPM, RMSSD = 2.0 ms, Muse β = 0.005.
4. **Usable** windows: no `artifact_flag`, Muse quality ≥ 0.75, Polar quality ≥ 0.85.
5. A **convergence window** has ≥ 2 of 3 physiological z-scores ≥ **2.0** and body-cam motion ≤ **0.15**. Consecutive convergence windows (1-window gap tolerance, ≥ 2 windows) merge into a segment.
6. For each segment the **audio lead** is measured: a transient that *precedes* physiological onset is consistent with a conventional startle.

Detected segments are matched to annotated events by temporal overlap. Investigator annotations and detector output stay separate evidence sources. Their agreement can raise priority, but an annotation can never force an event through a quality gate.

### §6 · Transparent event evidence score: gates before points
Recording quality is a **precondition**, not evidence.

**Gates** (a failed gate removes the event from candidacy, and the reason is recorded):
- session eligible for primary analysis
- signal quality not degraded and ≥ 50% of the event windows usable
- mean body-cam motion ≤ 0.20
- no documented environmental match

**Evidence points** (only for gate-passing events, averaged to 0–1):
- HR increase ≥ 5 BPM
- RMSSD decrease ≥ 15%
- β increase ≥ 8%
- window-level convergence confirmed by the independent detector
- independent audio support (0–1): reviewer found `voice_like` / `short_transient` / `ambiguous_noise` (+0.4), no possible environmental source (+0.3), Spirit Box off (+0.2), reviewer confidence ≥ 0.7 (+0.1); **zero** if the audio precedes the physiology by ≥ 10 s. An unreviewed event is *missing*, not zero.

**Penalties:** ×0.85 when the environmental status is *unknown* rather than *false*; ×0.5 for a startle timing pattern (audio lead ≥ 10 s).

Site properties (electricity, traffic, reputation, prior candidate counts) are **excluded** from the event score. The legacy v2 six-point score is kept only for comparison.

### §7 · Unsupervised anomaly detection
An **Isolation Forest** (400 trees, `random_state = 333`) sees only numerical features: HR/RMSSD/β deltas, audio peak and transients, motion, environmental noise, window max z-scores, usable fraction, and audio lead. No labels, no site reputation, no rule-based score. It answers *"which rows look unusual compared with the rest?"* and never *"which events are paranormal?"*

### §8 · Event triage score (0–100)
```
triage = 100 × (0.55 × evidence_score_v3 + 0.25 × ML anomaly percentile + 0.20 × audio support)
         × 0.5 if the event failed any gate (it stays visible but never dominates)
```
Weights are an explicit prioritization policy, not learned probabilities. Their robustness is tested in §13.

### §9 · Site-level repeatability (separate from the event score)
The site analysis counts **distinct participants**, not events, with at least one candidate (gate-passing event with evidence ≥ 0.6). Concentration is tested against a "site does not matter" null with a **hypergeometric test** and **Bonferroni correction** across sites, and uncorrected and corrected values are shown together. A **participant-level bootstrap** (5,000 resamples) gives uncertainty for each site's mean best triage score.

### §10 · Evidence records for LLM review
Each event becomes an auditable text record containing **observations only**: measurements, window statistics, timing, session conditions, the field log, and the independent audio review. Every score computed by the notebook, `explanation_status`, `review_note`, and all site summaries are **excluded**, and an assertion fails the run if a forbidden field leaks into a record. The LLM cannot echo a verdict it never saw.

### §11 · Evidence-bounded LLM review
- **System prompt rules:** never claim or imply paranormal causation; separate observation from interpretation; always name conventional explanations (movement, startle, radio bleed, equipment, weather, animals, infrastructure, signal quality); use only the record; treat "not reviewed" as missing, not negative.
- **Fixed question:** *"Does this event deserve additional human review?"* The model is never asked whether a place is haunted or what an event means.
- **Structured JSON schema:** `review_recommended`, `priority`, `confidence`, `strongest_evidence`, `conventional_explanations` (≥ 1 required), `limitations`, `values_cited`, all validated.
- **Consistency:** temperature 0, repeated `N_REPEATS` times; majority vote and self-agreement rate reported.
- **Invented-number check:** every number the model cites is matched against the record; unmatched numbers are counted as invented claims.
- **Review set:** the top `TOP_N` events **plus** `TOP_N` randomly sampled lower-ranked events, so the model cannot succeed by saying "yes" to everything.

**§11.1 Blind batch cross-check.** A second pass removes every field that carries a human or notebook judgement (`event_type`, field log, environmental match) and replaces site/participant IDs with salted SHA-1 hashes. The model returns candidate IDs; site aggregation is done in pandas, never by the model. This is a cross-check, **not validation**.

### §12 · Evaluation against generator ground truth (synthetic only)
Only now is `ground_truth.csv` loaded. Metrics: AUROC and precision@10/20/34 for every ranking, hard-negative leakage into the top 34, detector recall by label, LLM precision/recall on the review set, blind cross-check precision/recall, and whether the site test finds the independent-participant cluster while rejecting the single-participant and infrastructure clusters. **On real field data this section is replaced by a blinded replication study.**

### §13 · Weight-sensitivity analysis
2,000 plausible weight vectors are sampled from a Dirichlet distribution centred on (0.55, 0.25, 0.20). For each, the participant-level site score is recomputed and the median rank, % ranked first, and % in the top 3 are reported. A candidate that appears only under one precise weighting is **fragile**.

### §14 · Convergence table and export
The strongest candidates are shown with every analytical route side by side (rules, window detector, ML, audio, LLM, blind LLM, startle flag, gates). The goal is **convergence**, not a single score.

### §15 · Consolidated results, validation, and download
Executive summary, main results (re-read from the exported files), **visible limitations**, a dynamic export manifest that reopens every artifact and reports format, dimensions, fields, size, and status, and a timestamped ZIP of the entire run.

---

## 9. Reference run results (synthetic)

> ⚠️ **All numbers in this section are synthetic calibration results.** They show how the software behaves on a dataset whose answers are known. They are **not** evidence about any real place, person, or phenomenon.

Reference run `20260913_050900_UTC` · Deepnote · data source PostgreSQL/Neon (same content as the CSVs) · LLM mode **live** (`gemma4:latest`, temperature 0, 3 repeats).

### 9.1 Pipeline summary

| Metric | Result |
|---|---:|
| Events analyzed | 220 |
| Events passing all gates | 85 |
| Usable *during*-phase windows | 18,567 / 18,762 |
| Convergent windows | 536 |
| Detected physiological segments | 81 |
| Annotated events with an overlapping detected segment | 46 |
| Events with a startle timing pattern | 62 |
| Isolation Forest flags | 24 |
| Events reviewed by LLM | 40 (20 top + 20 random controls) |
| LLM review recommendations | 23 |

### 9.2 Ranking quality against designed labels

| Ranking | AUROC | precision@10 | precision@20 | precision@34 |
|---|---:|---:|---:|---:|
| v2 sensor score (legacy 6-point) | 0.903 | 0.40 | 0.65 | 0.53 |
| **v3 evidence score (gated)** | **0.960** | **1.00** | **0.95** | 0.76 |
| ML anomaly percentile (Isolation Forest alone) | 0.543 | 0.00 | 0.15 | 0.24 |
| **v3 triage score** | 0.935 | **1.00** | 0.90 | **0.82** |

### 9.3 Hard-negative leakage into the top 34

| Label | v2 sensor | v3 triage |
|---|---:|---:|
| `hn_env_confounded` | 8 | **0** |
| `hn_movement` | 4 | **0** |
| `hn_startle_lag` | 4 | 3 |
| `hn_spiritbox_radio` | 0 | 3 |
| `tp_full` | 16 | 16 |
| `tp_partial` | 1 | 6 |
| `tp_no_audio` | 1 | 6 |

The gates eliminate the environmental and movement traps. The remaining leakage (startle-lag and Spirit Box events) is reported, not hidden.

### 9.4 LLM reviewer behavior

| Check | Result |
|---|---:|
| Schema-valid responses | 120 / 120 |
| Mean self-agreement across repeats | 1.00 |
| Responses containing invented numbers | 0 / 40 |
| Agreement with triage split at median | 0.93 |
| Precision / recall on review set (synthetic) | 0.87 / 1.00 |
| Blind batch cross-check precision / recall (all 220) | 0.44 / 0.50 |

### 9.5 Site-level repeatability

| Site | Participants | Candidate participants | Mean best triage [95% bootstrap CI] | p (hypergeom.) | p (Bonferroni) | % ranked 1st (weights) | Designed cluster |
|---|---:|---:|---:|---:|---:|---:|---|
| SITE_14 | 10 | 5 | 55.6 [31.7, 79.0] | 0.008 | 0.120 | 100 | independent participants ✔ |
| SITE_15 | 8 | 3 | 38.8 [17.1, 63.6] | 0.108 | 1.000 | 0 | single participant |
| SITE_12 | 5 | 2 | 35.3 [8.6, 67.7] | 0.172 | 1.000 | 0 | isolated |
| SITE_09 | 8 | 1 | 28.3 [14.1, 47.6] | 0.755 | 1.000 | 0 | infrastructure (demoted ✔) |
| SITE_01 | 8 | 0 | 10.9 | 1.000 | 1.000 | 0 | infrastructure (demoted ✔) |

### 9.6 Visible limitations of this run

| Check | Result | Interpretation |
|---|---|---|
| Isolation Forest used alone | AUROC 0.543 | The unsupervised ranking is weak by itself and should remain a supporting signal |
| Blind LLM cross-check | precision 0.44, recall 0.50 | A cross-check, not a validation result |
| Top-site multiple-comparison correction | raw p 0.0080, Bonferroni p 0.1201 | Prioritize the site for replication; **not** confirmatory significance after correction |
| Ground truth | Synthetic generator labels | Performance does not establish paranormal causation or generalization to field data |

> A result that is notable before correction and unconvincing after it remains visible as **exploratory**, not as a confirmed effect.

---

## 10. Run package and audit trail

Each execution writes to `PSI_OUTPUT_DIR/output_<RUN_ID>/` and is zipped to `PSI_DOWNLOAD_DIR/psi_lab_v3_results_<RUN_ID>.zip`.

| Artifact | Content |
|---|---|
| `ranked_events_v3.csv` | Full event ranking: gates, evidence components, detector, ML, audio, LLM fields |
| `ranked_sites_v3.csv` | Participant-level site ranking with hypergeometric and Bonferroni p-values |
| `window_detected_segments.csv` | Physiological convergence segments from the window detector |
| `zscore_diagnostics_v3.csv` | Signal-specific baseline SDs, floors, and max z |
| `weight_sensitivity_v3.csv` | Site-rank stability across 2,000 weight vectors |
| `participant_bootstrap_v3.csv` | Participant-level bootstrap intervals by site |
| `presentation_top20_events_v3.csv` | Convergence table for the top 20 events |
| `evaluation_metrics.csv` | AUROC / precision@k against synthetic labels |
| `evidence_records.jsonl` | The exact observation-only record for every event |
| `llm_review_set_<mode>.csv` | Events and records sent to the reviewer |
| `llm_review_summary_<mode>.csv` | Majority vote, priority, confidence, agreement, invented-number count |
| `llm_raw_outputs_<mode>.json` | Every raw repeated LLM response (for audit) |
| `blind_llm_calls_<mode>.json`, `blind_llm_candidates_<mode>.csv` | Blind cross-check raw responses and candidates |
| `source_tables_csv/` | Portable snapshot of every input table |
| `run_summary.json` | Machine-readable executive summary |
| `pipeline_run_metadata.json` | Run ID, start/end time, duration, pipeline/prompt/dataset versions, seed, LLM mode, model, data source |
| `export_manifest.csv` | Every artifact reopened and validated (format, rows, columns, size, status) |

**Release levels (Appendix C of the paper):**

| Artifact | Typical release |
|---|---|
| Event ranking | Redacted or aggregated |
| Site analysis | Shareable when non-identifying |
| Detector diagnostics | Internal by default |
| Language-model audit | Summaries may be shared; raw records remain controlled |
| Synthetic evaluation | Shareable for calibration only |
| Run provenance | Shareable after secret and path review |

---

## 11. Stage II: prospective real-data field protocol

Stage II is **specified but not yet executed**. It defines how real data will be collected so that it can enter the same analytical schema.

### 11.1 Instruments and raw records
- **Muse 2** headset: raw EEG channels, contact-quality information, accelerometry.
- **Polar** heart-rate system with beat-to-beat RR export (exact model and firmware fixed before enrollment).
- One or more **body cameras** with original timestamped video retained.
- **Independent audio recorder** producing lossless or minimally compressed source files.
- **Environmental instruments**: temperature, humidity, ambient sound level, and any additional physical channel selected before preregistration.
- **Structured field log**: location, zone, investigator actions, known animals, occupancy, electrical infrastructure, running water, weather, traffic, deviations.

A **device registry** must be completed before collection: model, serial/asset ID, firmware, sampling rate, clock behavior, placement, calibration, software version, export format.

### 11.2 From physical signals to the analytical dataset
Raw recordings are **immutable**. Each native file is copied to a read-only archive, **hashed**, converted to a common UTC timeline (retaining original device timestamps), corrected for clock offset and drift, transformed under **frozen preprocessing rules**, and aggregated into non-overlapping **ten-second windows** with explicit missingness and quality flags. Derived CSVs are outputs of a documented transformation, not substitutes for the original evidence.

### 11.3 Session design
- **Standard session:** 5-min baseline → 20-min active period → 5-min recovery.
- **Target location:** selected *before* data collection for historical, testimonial, environmental, or investigative relevance.
- **Matched control location:** similar structure, access, acoustics, lighting, safety, movement constraints, and ordinary environmental sources. The control is a structured comparison condition, not an empty place.
- Equivalent activity scripts, investigator behavior, timing, sensor placement, and logging across conditions; **condition order randomized or counterbalanced**.

### 11.4 Repetition framework
- Each participant completes **two valid sessions per condition on separate days**.
- **Feasibility study:** 12 independent participants per target–control pair.
- **Confirmatory claims** require a *new* participant wave whose sample size is fixed by an **a priori power analysis**.

### 11.5 Session acceptance and exclusion
A session is valid for primary analysis only if preregistered requirements are met: valid consent, completed phases, required sensor streams, recoverable synchronization markers, acceptable residual alignment error, sufficient usable windows, documented protocol adherence, recorded physiological covariates. Event-level eligibility is separate from session validity: a valid session can contain an ineligible event, and one clean event cannot rescue an invalid session.

### 11.6 Primary outcome and confirmatory decision
A defensible primary outcome is the **participant-level proportion who produce at least one strong, gate-passing multimodal candidate in the target condition versus the matched control**. The participant, not the event, is the experimental unit.

A site result is methodologically supported only when **all** of the following hold:
1. the preregistered target–control contrast passes its inferential criterion in the **confirmatory** wave;
2. the direction is consistent across collection sessions;
3. no single participant drives the result;
4. quality and conventional-explanation gates remain satisfied;
5. the result does not depend on one narrow scoring-weight choice.

Even then, the conclusion is *a replicated unexplained multimodal pattern under study conditions*, **not a paranormal cause**.

### 11.7 Real-data session checklist (Appendix B)

- **Before collection:** consent complete; participant code assigned; condition order controlled; device registry and firmware verified; storage available; clocks checked; sensor placement documented; scripted control markers recorded.
- **During collection:** baseline complete; activity script followed; zone and intervention markers recorded; operator movement documented; conventional sources logged; participant stop status monitored.
- **After collection:** closing synchronization marker recorded; device status checked; participant debriefed; native files copied to raw archive; hashes calculated; deviations recorded before analysis; target/control labels protected from blinded reviewers.
- **Before primary analysis:** transformation version fixed; alignment and completeness checks passed; exclusions applied without outcome access; audio review complete; code, prompts, weights, and primary endpoint frozen.

---

## 12. Preregistration, governance, and ethics

- **Before the pilot:** freeze acquisition schema, quality rules, exclusions, detector logic, scoring weights, LLM prompt, primary endpoint, and planned summaries.
- **Before confirmation:** preregister the revised protocol with timestamp, sample-size justification, stopping rule, multiplicity plan, and a clear distinction between confirmatory and exploratory analyses (for example on [OSF Registries](https://osf.io/registries)).
- **Sensitive data:** physiological and EEG data are sensitive personal data. Participants receive plain-language information about each sensor, recording duration, foreseeable discomfort, data use, retention, and withdrawal. Identifiers are stored separately from analytical data; publication files use anonymized codes and redacted notes.
- **Release levels:** the synthetic package may include complete input snapshots because it contains no real participants. Real-data packages require at least two release levels: an **internal audit package** for authorized personnel, and a **collaborator/publication package** with approved aggregates, redacted event summaries, methods, limitations, and non-identifying metadata.

---

## 13. Limitations and transition criteria

**Limitations.** The software has been exercised only on synthetic data whose patterns were designed by the generator. Success on that bench can reveal implementation defects and analytical blind spots, but it may **overestimate** performance, because the real world contains artifacts and dependencies the generator did not imagine. Consumer EEG is vulnerable to motion, muscle activity, poor electrode contact, and interference. HRV depends on beat-detection quality and physiological context. Video and audio features can be driven by operator movement, gain control, compression, reflections, radio sources, animals, plumbing, weather, and infrastructure. Participants bring expectations, fatigue, medication, caffeine, prior exertion, and emotional response. These are part of the causal structure a serious protocol must record.

**Ready to leave the synthetic-only stage only after:**
- [ ] the device registry is complete;
- [ ] engineering runs achieve **three consecutive passes**;
- [ ] transformation code is verified against scripted events;
- [ ] consent and data-governance materials are approved;
- [ ] pilot rules are frozen;
- [ ] the target–control design is preregistered.

**Ready for confirmatory language only after** an independent participant wave is powered and completed under a frozen primary analysis.

The formal sequence is: **calibrate** the software on known synthetic cases → **validate** acquisition and synchronization through engineering runs → run a matched, counterbalanced **feasibility pilot** → **preregister and power** a confirmatory wave.

> **Operational principle:** *Investigate first. Conclude afterward.*

---

## 14. Security and secrets

- Real credentials live only in `paranormal/.env`, which is **excluded by `.gitignore`** (`paranormal/.env` and `**/.env`). Only `paranormal/.env.example`, which contains placeholders, is versioned.
- The notebook prints only whether a key or URL is *configured* (`True`/`False`), never its value.
- SQL identifiers are validated against a strict pattern before use.
- Before sharing a run package, review `pipeline_run_metadata.json` for local paths (Appendix C: "shareable after secret and path review").
- If a secret is ever committed by mistake, **rotate it immediately**. Deleting the file in a later commit does not remove it from Git history.

---

## 15. Archiving on Zenodo, OSF Preprints, and ResearchGate

This repository is prepared for deposit as a citable, versioned research object.

**Zenodo (recommended for the DOI of record)**
1. Log in to [zenodo.org](https://zenodo.org) with GitHub and enable this repository under *Account → GitHub*.
2. Create a GitHub **release** (for example tag `v1.0`). Zenodo archives the release automatically and mints a DOI, reading authorship and metadata from `CITATION.cff`.
3. Add the DOI badge to the top of this README and the DOI to `CITATION.cff` and the paper's citation table.
4. Suggested Zenodo resource type: *Publication → Preprint* (paper) or *Software* (repository), license **CC BY-NC-ND 4.0**.

**OSF Preprints**
Upload `PSI_Field_Lab_Methodology_Paper_v1.0.pdf` as the preprint file, link this GitHub repository as supplemental material, and use the abstract and keywords below. Use OSF Registries for the Stage II preregistration when it is ready.

**ResearchGate**
Add the paper as a *Preprint* or *Research proposal*, attach the PDF, and link the Zenodo DOI and this repository.

**Ready-to-paste metadata**

- **Title:** PSI Field Lab: Synthetic Calibration and Prospective Field Validation Protocol
- **Author:** Alberto Cardenas, Paranormal 3:33 Research Lab
- **Version / date:** 1.0 / September 2026
- **License:** CC BY-NC-ND 4.0
- **Keywords:** paranormal research; parapsychology; synthetic calibration; field validation; multimodal sensors; Muse 2; heart-rate variability; anomaly triage; preregistration; repeatability; language-model review; research ethics
- **Abstract:** This paper presents the PSI Field Lab as a two-stage research methodology for documenting, synchronizing, auditing, and reviewing multimodal field observations in anomalous and paranormal research contexts. Stage I is a synthetic calibration demonstration. Its records were generated computationally with a fixed seed and a documented experimental structure designed to resemble the derived data that a future field study could produce. It tests data integration, quality gates, transparent scoring, unsupervised anomaly detection, site-level repeatability, sensitivity analysis, constrained language-model review, and run-level audit packaging. It does not contain real sensor recordings or evidence about paranormal phenomena. Stage II is a prospective real-data protocol. It specifies how physiological, EEG, motion, video, audio, environmental, and contextual records would be collected using a Muse 2 headset, a Polar heart-rate system capable of exporting beat-to-beat intervals, body cameras, an independent audio recorder, and environmental instruments. It defines clock synchronization, immutable raw-data retention, transformation into a common ten-second analytical schema, matched target and control sessions, randomized order, blinded review, quality gates, exclusions, and repetition requirements. Across both stages, the system is restricted to documentation and evidence triage. A high score means unusual under recorded conditions, not paranormal. The methodological contribution is a traceable bridge from synthetic software calibration to preregistered field acquisition, with observation, interpretation, explanation, and causal claims kept separate.

**Statement for collaborators (Appendix D).** The PSI Field Lab currently demonstrates an analytical workflow using synthetic, sensor-informed data. It does not present observations collected from real participants or physical sites. The prospective field protocol is designed to test whether predefined multimodal patterns repeat across target and matched control conditions under fixed acquisition, quality, blinding, and analysis rules. Any resulting event remains a candidate for human review, not proof of paranormal causation.

---

## 16. How to cite

> Cardenas, A. (2026). *PSI Field Lab: Synthetic Calibration and Prospective Field Validation Protocol*. Paranormal 3:33 Research Lab. Version 1.0. https://github.com/betoalien/Paranormal-333-PSI-Field-Lab

```bibtex
@techreport{cardenas2026psifieldlab,
  author      = {Cardenas, Alberto},
  title       = {{PSI Field Lab}: Synthetic Calibration and Prospective Field Validation Protocol},
  institution = {Paranormal 3:33 Research Lab},
  year        = {2026},
  month       = sep,
  type        = {Methodology paper},
  number      = {Version 1.0},
  url         = {https://github.com/betoalien/Paranormal-333-PSI-Field-Lab},
  note        = {DOI pending}
}
```

GitHub's **"Cite this repository"** button (generated from `CITATION.cff`) provides APA and BibTeX formats.

**Intellectual priority and versioning.** This work establishes the methodological framework, terminology, staged validation structure, analytical workflow, and governance assumptions of the PSI Field Lab as developed by Alberto Cardenas and Paranormal 3:33 Research Lab. Future work that uses, adapts, implements, or extends this framework should cite this version or the corresponding repository version of record. This statement does not restrict independent research into anomalous or paranormal claims. It preserves a clear record of this specific data-centered methodology and its authorship.

---

## 17. License

This work is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International** license (CC BY-NC-ND 4.0). See [`LICENSE`](LICENSE) or <https://creativecommons.org/licenses/by-nc-nd/4.0/>.

You may share the material with attribution for non-commercial purposes. Distributing modified versions requires the author's permission. You may still run the code and regenerate the data to reproduce the results. For collaboration or adaptation requests, see [Contact](#19-contact).

---

## 18. References

- Bem, D. J. (2011). Feeling the future: Experimental evidence for anomalous retroactive influences on cognition and affect. *Journal of Personality and Social Psychology, 100*(3), 407–425.
- Brysbaert, M. (2019). How many participants do we have to include in properly powered experiments? *Journal of Cognition, 2*(1), 16.
- Duane, T. D., & Behrendt, T. (1965). Extrasensory electroencephalographic induction between identical twins. *Science, 150*, 367.
- Grinberg-Zylberbaum, J., Delaflor, M., Attie, L., & Goswami, A. (1994). The Einstein-Podolsky-Rosen paradox in the brain: The transferred potential. *Physics Essays, 7*(4), 422–428.
- Grinberg-Zylberbaum, J., & Ramos, J. (1987). Patterns of interhemispheric correlation during human communication. *International Journal of Neuroscience, 36*(1–2), 41–55.
- Lakens, D. (2022). Sample size justification. *Collabra: Psychology, 8*(1), 33267.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606.
- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology. *Psychological Science, 22*(11), 1359–1366.
- Targ, R., & Puthoff, H. E. (1974). Information transmission under conditions of sensory shielding. *Nature, 251*, 602–607.
- Wagenmakers, E.-J., Wetzels, R., Borsboom, D., & van der Maas, H. L. J. (2011). Why psychologists must change the way they analyze their data: The case of psi. *Journal of Personality and Social Psychology, 100*(3), 426–432.
- Wiseman, R., & Watt, C. (2006). Belief in psychic ability and the misattribution hypothesis. *British Journal of Psychology, 97*(3), 323–338.

---

## 19. Contact

**Alberto Cardenas**, Founder, Paranormal 3:33 Research Lab
✉️ iam@albertocardenas.com

*For Paranormal 3:33 Research Lab, the central contribution is a move from stories to structured evidence, and from a convincing demonstration to a protocol capable of being tested, audited, and repeated.*
