# PSI Field Lab — synthetic dataset v2

Generated with `generate_psi_data_v2.py --seed 333`. Drop-in replacement for the
`paranormal/` folder used by `PSI_Lab_v2_Presentation_Ready.ipynb`: same columns,
dtypes, vocabularies and hard-coded IDs (`SITE_14_P01_A`, `EVT_0001`, `EVT_0015`,
`EVT_0043`, 220 events). The v2 notebook runs unchanged; the v3 notebook
(`PSI_Lab_v3_LLM_Ready.ipynb`) uses `ground_truth.csv` for evaluation only.

## Files
| file | rows | notes |
|---|---|---|
| `sessions.csv` | 162 | 150 trial A + 12 trial-B repeats of interrupted sessions |
| `sensor_windows.csv` | 28,122 | 10-second windows; events are derived from here |
| `events.csv` | 220 | deltas computed against the 60 s preceding each event |
| `audio_reviews.csv` | 160 | coverage by protocol: every event with audio + 20% random controls |
| `ground_truth.csv` | 220 | **NEW.** Design labels. Evaluation only — never feed it to scoring or the LLM |

## Labels in `ground_truth.csv`
Designed positives (`is_designed_anomaly = True`, 34):
- `tp_full` (16): three physiological modalities + audio, low motion, no environmental explanation.
- `tp_partial` (12): two of three modalities, values close to the notebook thresholds.
- `tp_no_audio` (6): physiological convergence; the independent recorder captures nothing.

Hard negatives (`hn_*`, 48) — all of them should stay OUT of the top ranks:
- `hn_movement` (12): full physiology explained by a movement burst (motion 0.30–0.50).
- `hn_degraded_quality` (8): beta/HR spike with Muse quality 0.45–0.65 and `artifact_flag`.
- `hn_spiritbox_radio` (8): `voice_like` audio with the Spirit Box ON; mild physiology.
- `hn_env_confounded` (12): full convergence during a documented electrical interval. 5 at SITE_01 and 4 at SITE_09 form false infrastructure clusters.
- `hn_startle_lag` (8): the transient occurs 10 s BEFORE the physiology. Only distinguishable in `sensor_windows`; in `events.csv` it looks like a `tp_full`.

Background (`bg_*`, 126) and `protocol_interruption` (12, in ineligible sessions).

## Site structure (`designed_cluster`)
- `SITE_14_independent_participants`: positives across 8 different participants → genuine repeatability.
- `SITE_15_single_participant`: 3 of 4 positives in the SAME participant → not repeatability.
- `SITE_01` / `SITE_09_infrastructure_cluster`: look "hot" until context is applied.
- `isolated`: single positives at sites with no repetition.

## Baseline: the unchanged v2 notebook on this data
- `sensor_evidence_score` is no longer bimodal: 0→4, 1→17, 2→62, 3→52, 4→31, 5→27, 6→27.
- 54 events score ≥ 5; only 34 are designed positives. `hn_env_confounded`, `hn_movement`
  and `hn_startle_lag` reach 5–6.
- Triage precision vs. ground truth: @10 = 1.00, @20 = 0.85, @34 = 0.59.
- The sensor-only permutation test picks SITE_01 (infrastructure cluster) as the strongest
  site, with p = 1.0.
- SITE_14 stays #1, but SITE_09 (infrastructure) ranks #4 with 5 "strong participants".

These are the numbers the v3 pipeline has to beat.

## Reproducibility
- Single seed (`--seed`); only numpy/pandas required.
- A different seed gives a new realization of the same design — useful for measuring
  pipeline variance across datasets.
