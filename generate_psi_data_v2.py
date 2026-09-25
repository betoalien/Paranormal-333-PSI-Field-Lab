"""
PSI Field Lab -- Synthetic dataset generator v2
Paranormal 3:33 Research Lab

Generates the 4 CSV tables consumed by `PSI_Lab_v2_Presentation_Ready.ipynb`
with EXACTLY the same schema (columns, dtypes, vocabularies, IDs), plus a fifth
file `ground_truth.csv` holding the design labels used to evaluate the pipeline.
The notebook never reads `ground_truth.csv`, so nothing leaks.

What changes relative to dataset v1:
  * Realistic physiology: HR sd ~3-4 bpm per window, slow AR(1) drift, a
    "during" phase with walking and movement bursts that raise HR.
  * Events are DERIVED from sensor_windows (deltas computed from the windows),
    not the other way around.
  * Positives with graded difficulty: full, partial (2 of 3 modalities, near
    threshold) and without audio corroboration.
  * Hard negatives: movement, degraded signal quality, Spirit Box ON with a
    "voice_like" result, documented environmental interval with full
    convergence, and a startle with lag (transient 10 s BEFORE the physiology:
    only detectable in the windows).
  * Structured site clusters: SITE_14 = positives across independent
    participants; SITE_15 = positives concentrated in ONE participant;
    SITE_01 and SITE_09 = clusters confounded by infrastructure.
  * audio_reviews coverage defined by protocol (every event with audio + a
    random control sample), not by event type.

Usage:
    python generate_psi_data_v2.py --out paranormal_v2 --seed 333
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reference schema (copied from the v1 CSVs). The generator guarantees this
# exact column order.
# --------------------------------------------------------------------------
SESSIONS_COLS = [
    "session_id", "site_id", "site_name", "participant_id", "trial",
    "repeat_of_session_id", "session_date", "start_time_local",
    "planned_duration_minutes", "actual_duration_minutes", "session_status",
    "primary_analysis_eligible", "bodycam_status", "muse2_status",
    "polar_status", "recording_sync_status", "building_type",
    "construction_material", "occupancy_status", "electricity_available",
    "running_water_available", "traffic_level", "cellular_signal",
    "temperature_c", "humidity_pct", "weather_condition", "rainfall_mm",
    "wind_kph", "ambient_noise_db", "known_animals_observed",
    "known_electrical_sources", "protocol_deviation", "investigator_notes",
]
WINDOWS_COLS = [
    "session_id", "window_id", "window_start_sec", "window_end_sec", "phase",
    "investigation_zone", "heart_rate_bpm", "rr_mean_ms", "rmssd_ms",
    "sdnn_ms", "muse_delta_rel", "muse_theta_rel", "muse_alpha_rel",
    "muse_beta_rel", "muse_gamma_rel", "muse_accel_mean_g",
    "muse_accel_peak_g", "bodycam_motion_score",
    "bodycam_light_change_score", "audio_rms_dbfs", "audio_peak_dbfs",
    "audio_transient_count", "temperature_c", "humidity_pct",
    "environmental_noise_db", "polar_signal_quality", "muse_signal_quality",
    "bodycam_signal_quality", "artifact_flag",
]
EVENTS_COLS = [
    "event_id", "session_id", "site_id", "participant_id", "event_start_sec",
    "event_end_sec", "phase", "investigation_zone", "event_type",
    "source_modalities", "detection_method", "observation_text",
    "explanation_status", "known_explanation", "environmental_match",
    "manual_marker", "signal_quality_summary", "hr_delta_bpm_vs_pre_event",
    "rmssd_delta_pct_vs_pre_event", "beta_delta_pct_vs_pre_event",
    "audio_peak_dbfs", "audio_transient_count", "bodycam_motion_score",
    "environmental_noise_db", "review_note",
]
AUDIO_COLS = [
    "audio_review_id", "event_id", "session_id", "site_id", "participant_id",
    "review_start_sec", "review_end_sec", "recorder_status",
    "spirit_box_status", "audio_result", "audible_content",
    "speech_intelligibility", "radio_bleed_possible",
    "environmental_source_possible", "reviewer_confidence",
    "explanation_status", "review_notes",
]

# --------------------------------------------------------------------------
# Sites (same 15 sites as v1 so the narrative and hard-coded IDs still work)
# --------------------------------------------------------------------------
SITES = [
    # id, name, building_type, material, occupancy, elec, water, traffic, cell, animals, elec_sources, base_noise
    ("SITE_01", "Mercer Historic House", "historic_house", "wood_brick", "unoccupied", True, True, "moderate", "good", "none_observed", "old_wiring,refrigerator,exterior_transformer", 40),
    ("SITE_02", "Riverside Motor Hotel", "old_hotel", "brick_concrete", "partially_active", True, True, "moderate", "good", "none_observed", "hvac,elevator_panel,vending_machines", 44),
    ("SITE_03", "Oak Ridge Cemetery", "cemetery", "open_ground_stone", "outdoor", False, False, "low", "good", "none_observed", "nearby_road_lighting", 36),
    ("SITE_04", "North Foundry Warehouse", "industrial_warehouse", "steel_concrete", "inactive", True, False, "low", "fair", "none_observed", "breaker_panels,nearby_power_lines", 42),
    ("SITE_05", "Jefferson Rural School", "abandoned_school", "brick_wood", "abandoned", False, False, "very_low", "fair", "observed_outside_primary_window", "none_on_site", 33),
    ("SITE_06", "Grand Lyric Theater", "historic_theater", "brick_wood_plaster", "inactive", True, True, "moderate", "good", "none_observed", "stage_power,hvac,emergency_lighting", 43),
    ("SITE_07", "St. Agnes Historic Chapel", "historic_chapel", "stone_wood", "inactive", True, True, "low", "fair", "none_observed", "lighting_panel,audio_system", 37),
    ("SITE_08", "Briar House Basement", "residential_basement", "concrete_brick", "occupied_building", True, True, "low", "poor", "none_observed", "sump_pump,furnace,water_heater", 41),
    ("SITE_09", "Atlas Textile Mill", "former_factory", "brick_steel", "inactive", True, False, "moderate", "good", "none_observed", "adjacent_substation,security_lighting", 44),
    ("SITE_10", "East Quarry Service Tunnel", "stone_tunnel", "stone_concrete", "inactive", False, False, "very_low", "none", "none_observed", "none_on_site", 30),
    ("SITE_11", "Cedar Creek Ranch House", "rural_ranch_house", "wood_stone", "seasonally_used", True, True, "very_low", "fair", "none_observed", "well_pump,refrigerator,exterior_lighting", 35),
    ("SITE_12", "County Heritage Museum", "museum", "brick_concrete", "active_daytime", True, True, "moderate", "good", "none_observed", "security_system,hvac,display_power,wifi", 42),
    ("SITE_13", "Maple Street Victorian", "victorian_house", "wood_brick", "unoccupied", True, True, "low", "good", "none_observed", "old_wiring,water_heater,street_transformer", 39),
    ("SITE_14", "San Aurelio Stone Hacienda", "abandoned_stone_hacienda", "stone_masonry", "abandoned", False, False, "very_low", "poor", "none_observed", "none_on_site", 31),
    ("SITE_15", "Las Palmas Rural Ruins", "isolated_historic_ruins", "stone_adobe", "abandoned", False, False, "very_low", "poor", "none_observed", "none_on_site", 30),
]
SITE_NOTES = {
    "SITE_01": "Old wiring and active plumbing documented before the study.",
    "SITE_02": "HVAC cycling, plumbing and occasional hallway activity are expected.",
    "SITE_03": "Wind, insects and distant road traffic are recurring sound sources.",
    "SITE_04": "Metal structure and nearby electrical infrastructure produce hum and expansion noise.",
    "SITE_05": "Birds, small animals and loose windows were observed on site.",
    "SITE_06": "Stage rigging and HVAC produce intermittent creaks and cycling sounds.",
    "SITE_07": "Stone walls create strong echo; audio system was unplugged for the study.",
    "SITE_08": "Sump pump and furnace cycle during the night; occupants upstairs.",
    "SITE_09": "Adjacent substation hum is audible throughout the building.",
    "SITE_10": "Long tunnel with wind echo; no infrastructure on site.",
    "SITE_11": "Well pump and refrigerator cycle intermittently; livestock nearby.",
    "SITE_12": "Security system and display power remain active at night.",
    "SITE_13": "Old wiring and street transformer documented; water heater cycles.",
    "SITE_14": "No infrastructure on site; wind through stone corridors documented.",
    "SITE_15": "No infrastructure on site; open ruins exposed to wind.",
}
DEVIATIONS = [
    "rain", "rain", "building_maintenance_noise", "animal_intrusion", "staff_entry",
    "street_noise", "sensor_sync_failure", "electrical_maintenance",
    "water_intrusion", "livestock_intrusion", "security_alarm_test", "plumbing_activity",
]

WINDOW_SEC = 10
N_BEFORE, N_DURING, N_AFTER = 30, 120, 30  # 5 + 20 + 5 minutes
ZONES = ["zone_a", "zone_b", "zone_c", "zone_d"]


def zone_for(start_sec: int) -> str:
    return ZONES[min(3, (start_sec - 300) // 300)]


# --------------------------------------------------------------------------
# 1. Sessions
# --------------------------------------------------------------------------
def make_sessions(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    date0 = pd.Timestamp("2026-03-02")
    interrupted = set(rng.choice(150, size=12, replace=False))
    deviations = list(DEVIATIONS)
    rng.shuffle(deviations)
    k = 0
    idx = 0
    for si, site in enumerate(SITES):
        sid, name, btype, mat, occ, elec, water, traffic, cell, animals, esrc, base_noise = site
        site_date = date0 + pd.Timedelta(days=si * 3)
        temp0 = rng.uniform(16, 26)
        hum0 = rng.uniform(40, 70)
        weather = rng.choice(["clear", "partly_cloudy", "rain"], p=[0.88, 0.09, 0.03])
        for p in range(1, 11):
            pid = f"P{p:02d}"
            base = dict(
                site_id=sid, site_name=name, participant_id=pid,
                planned_duration_minutes=30, building_type=btype,
                construction_material=mat, occupancy_status=occ,
                electricity_available=elec, running_water_available=water,
                traffic_level=traffic, cellular_signal=cell,
                known_animals_observed=animals, known_electrical_sources=esrc,
            )
            is_int = idx in interrupted
            idx += 1
            dur = int(rng.integers(8, 21)) if is_int else 30
            dev = deviations[k] if is_int else np.nan
            if is_int:
                k += 1
            w = "rain" if dev == "rain" else weather
            rows.append(dict(
                base, session_id=f"{sid}_{pid}_A", trial="A", repeat_of_session_id=np.nan,
                session_date=site_date.strftime("%Y-%m-%d"),
                start_time_local=(pd.Timestamp("18:30") + pd.Timedelta(minutes=40 * (p - 1))).strftime("%H:%M"),
                actual_duration_minutes=dur,
                session_status="interrupted" if is_int else "completed",
                primary_analysis_eligible=not is_int,
                bodycam_status="partial" if is_int else "complete",
                muse2_status="partial" if (is_int or rng.random() < 0.04) else "complete",
                polar_status="complete",
                recording_sync_status="partial" if dev == "sensor_sync_failure" else "synced",
                temperature_c=round(temp0 + rng.normal(0, 0.8), 1),
                humidity_pct=round(hum0 + rng.normal(0, 2.5), 1),
                weather_condition=w,
                rainfall_mm=round(rng.uniform(0.5, 4.0), 1) if w == "rain" else 0.0,
                wind_kph=round(rng.uniform(2, 18), 1),
                ambient_noise_db=round(base_noise + rng.normal(0, 2.5), 1),
                protocol_deviation=dev,
                investigator_notes=(
                    f"{SITE_NOTES[sid]} Session interrupted ({dev.replace('_', ' ')}); repeated as trial B."
                    if is_int else f"{SITE_NOTES[sid]} Protocol completed without deviation."
                ),
            ))
            if is_int:
                rows.append(dict(
                    base, session_id=f"{sid}_{pid}_B", trial="B",
                    repeat_of_session_id=f"{sid}_{pid}_A",
                    session_date=(site_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    start_time_local="19:00", actual_duration_minutes=30,
                    session_status="completed", primary_analysis_eligible=True,
                    bodycam_status="complete", muse2_status="complete", polar_status="complete",
                    recording_sync_status="synced",
                    temperature_c=round(temp0 + rng.normal(0, 0.8), 1),
                    humidity_pct=round(hum0 + rng.normal(0, 2.5), 1),
                    weather_condition=weather,
                    rainfall_mm=0.0, wind_kph=round(rng.uniform(2, 18), 1),
                    ambient_noise_db=round(base_noise + rng.normal(0, 2.5), 1),
                    protocol_deviation=np.nan,
                    investigator_notes=f"{SITE_NOTES[sid]} Repeat of interrupted session; protocol completed without deviation.",
                ))
    df = pd.DataFrame(rows)[SESSIONS_COLS]
    df = df.sort_values("session_id").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2. Sensor windows (realistic physiology)
# --------------------------------------------------------------------------
def ar1(rng, n, sd, rho=0.85):
    x = np.zeros(n)
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def make_windows(rng: np.random.Generator, sessions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    site_noise = {s[0]: s[11] for s in SITES}
    for _, s in sessions.iterrows():
        n_total = int(s.actual_duration_minutes * 60 / WINDOW_SEC)
        n = np.arange(n_total)
        start = n * WINDOW_SEC
        phase = np.where(start < 300, "before", np.where(start < 1500, "during", "after"))
        zone = np.array([
            "baseline_station" if p == "before" else ("recovery_station" if p == "after" else zone_for(t))
            for p, t in zip(phase, start)
        ])
        during = phase == "during"

        # --- participant-specific baseline
        hr0 = rng.normal(68, 5)
        rmssd0 = rng.normal(42, 9)
        beta0 = rng.normal(0.20, 0.02)

        # --- motion: slow walking in "during" + bursts
        motion = np.clip(rng.gamma(2.0, 0.035, n_total) + during * 0.07, 0, 1)
        n_bursts = rng.integers(2, 6)
        for _ in range(n_bursts):
            i = rng.integers(30, max(31, n_total - 3))
            L = rng.integers(1, 4)
            motion[i:i + L] += rng.uniform(0.15, 0.35)
        motion = np.clip(motion, 0.005, 0.95)
        accel_mean = np.clip(0.02 + 0.25 * motion + rng.normal(0, 0.01, n_total), 0.005, 1)
        accel_peak = np.clip(accel_mean * rng.uniform(2.5, 4.5, n_total), 0.02, 3)

        # --- HR: baseline + AR(1) drift + walking + motion coupling + noise
        hr = hr0 + ar1(rng, n_total, 2.0) + during * 3.0 + 22 * (motion - 0.1).clip(0) + rng.normal(0, 2.2, n_total)
        rr = 60000 / hr
        rmssd = np.clip(rmssd0 + ar1(rng, n_total, 5.0) - 40 * (motion - 0.1).clip(0) + rng.normal(0, 5.0, n_total), 8, 150)
        sdnn = np.clip(rmssd * rng.uniform(1.2, 1.6) + rng.normal(0, 4, n_total), 10, 200)

        # --- relative EEG bands (sum to 1)
        beta = beta0 + ar1(rng, n_total, 0.012) + 0.04 * (motion - 0.1).clip(0) + rng.normal(0, 0.012, n_total)
        delta = 0.29 + rng.normal(0, 0.02, n_total)
        theta = 0.20 + rng.normal(0, 0.015, n_total)
        alpha = 0.25 + rng.normal(0, 0.02, n_total)
        gamma = 0.05 + rng.normal(0, 0.008, n_total)
        eeg = np.clip(np.vstack([delta, theta, alpha, beta, gamma]), 0.01, None)
        eeg = eeg / eeg.sum(axis=0)

        # --- audio and environment
        env_noise = site_noise[s.site_id] + ar1(rng, n_total, 2.5) + rng.normal(0, 1.5, n_total)
        rms = -47 + 0.25 * (env_noise - 40) + rng.normal(0, 1.5, n_total)
        peak = rms + rng.uniform(9, 15, n_total)
        transients = rng.poisson(0.03 + 0.15 * motion, n_total)
        peak = peak + 6 * (transients > 0)
        light = np.clip(rng.gamma(1.5, 0.04, n_total) + 0.02 * motion, 0, 1)

        # --- signal quality
        pq = np.clip(0.96 - 0.15 * (motion - 0.15).clip(0) + rng.normal(0, 0.015, n_total), 0.5, 1)
        mq = np.clip(0.94 - 0.30 * (motion - 0.15).clip(0) + rng.normal(0, 0.03, n_total), 0.3, 1)
        bq = np.clip(0.98 + rng.normal(0, 0.01, n_total), 0.85, 1)
        artifact = (mq < 0.75) | (pq < 0.85)

        temp = s.temperature_c + ar1(rng, n_total, 0.3)
        hum = s.humidity_pct + ar1(rng, n_total, 0.8)

        frames.append(pd.DataFrame({
            "session_id": s.session_id,
            "window_id": [f"{s.session_id}_W{i + 1:03d}" for i in n],
            "window_start_sec": start, "window_end_sec": start + WINDOW_SEC,
            "phase": phase, "investigation_zone": zone,
            "heart_rate_bpm": hr, "rr_mean_ms": rr, "rmssd_ms": rmssd, "sdnn_ms": sdnn,
            "muse_delta_rel": eeg[0], "muse_theta_rel": eeg[1], "muse_alpha_rel": eeg[2],
            "muse_beta_rel": eeg[3], "muse_gamma_rel": eeg[4],
            "muse_accel_mean_g": accel_mean, "muse_accel_peak_g": accel_peak,
            "bodycam_motion_score": motion, "bodycam_light_change_score": light,
            "audio_rms_dbfs": rms, "audio_peak_dbfs": peak, "audio_transient_count": transients,
            "temperature_c": temp, "humidity_pct": hum, "environmental_noise_db": env_noise,
            "polar_signal_quality": pq, "muse_signal_quality": mq, "bodycam_signal_quality": bq,
            "artifact_flag": artifact,
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 3. Event plan (design labels) and injection into the windows
# --------------------------------------------------------------------------
def event_plan(rng: np.random.Generator, sessions: pd.DataFrame) -> list[dict]:
    """Return the list of events to inject: (session_id, label, cluster)."""
    elig = sessions[sessions.primary_analysis_eligible].copy()
    by_site = {s: g.session_id.tolist() for s, g in elig.groupby("site_id")}
    plan = []

    def pick(site, n, exclude=()):
        pool = [x for x in by_site[site] if x not in exclude]
        return list(rng.choice(pool, size=n, replace=False))

    # --- Full positives
    # SITE_14: 6 independent participants -> genuine cluster
    for sid in pick("SITE_14", 6):
        plan.append(dict(session_id=sid, label="tp_full", cluster="SITE_14_independent_participants"))
    # SITE_15: 4 events but 3 in the SAME participant (not independent)
    s15 = pick("SITE_15", 2)
    plan += [dict(session_id=s15[0], label="tp_full", cluster="SITE_15_single_participant")] * 3
    plan.append(dict(session_id=s15[1], label="tp_full", cluster="SITE_15_single_participant"))
    # isolated positives at other sites (no repeatability)
    for site in ["SITE_03", "SITE_05", "SITE_07", "SITE_09", "SITE_11", "SITE_12"]:
        plan.append(dict(session_id=pick(site, 1)[0], label="tp_full", cluster="isolated"))

    # --- Partial positives (12): 2 of 3 modalities, near threshold
    for site in ["SITE_14", "SITE_14", "SITE_05", "SITE_05", "SITE_09", "SITE_11",
                 "SITE_03", "SITE_13", "SITE_02", "SITE_10", "SITE_06", "SITE_15"]:
        plan.append(dict(session_id=pick(site, 1)[0], label="tp_partial", cluster="isolated" if site not in ("SITE_14",) else "SITE_14_independent_participants"))

    # --- Positives without audio (6): physiological convergence, recorder detects nothing
    for site in ["SITE_14", "SITE_15", "SITE_04", "SITE_08", "SITE_10", "SITE_12"]:
        plan.append(dict(session_id=pick(site, 1)[0], label="tp_no_audio", cluster="isolated"))

    # --- Hard negatives
    # movement (12): HR/RMSSD/beta respond to a movement burst + footsteps
    for site in rng.choice([s[0] for s in SITES], 12):
        plan.append(dict(session_id=pick(site, 1)[0], label="hn_movement", cluster="none"))
    # degraded quality (8): beta/HR spike with Muse quality ~0.5
    for site in rng.choice([s[0] for s in SITES], 8):
        plan.append(dict(session_id=pick(site, 1)[0], label="hn_degraded_quality", cluster="none"))
    # Spirit Box ON (8): "voice_like" audio with radio sweep, mild physiology
    for site in ["SITE_14", "SITE_15", "SITE_05", "SITE_10", "SITE_03", "SITE_07", "SITE_11", "SITE_13"]:
        plan.append(dict(session_id=pick(site, 1)[0], label="hn_spiritbox_radio", cluster="none"))
    # environmentally confounded (12): FULL convergence during a documented electrical interval
    for sid in pick("SITE_01", 5):
        plan.append(dict(session_id=sid, label="hn_env_confounded", cluster="SITE_01_infrastructure_cluster"))
    for sid in pick("SITE_09", 4):
        plan.append(dict(session_id=sid, label="hn_env_confounded", cluster="SITE_09_infrastructure_cluster"))
    for site in ["SITE_04", "SITE_08", "SITE_13"]:
        plan.append(dict(session_id=pick(site, 1)[0], label="hn_env_confounded", cluster="none"))
    # startle with lag (8): transient 10 s BEFORE the physiology
    for site in ["SITE_14", "SITE_15", "SITE_10", "SITE_05", "SITE_03", "SITE_06", "SITE_02", "SITE_12"]:
        plan.append(dict(session_id=pick(site, 1)[0], label="hn_startle_lag", cluster="none"))

    # --- Protocol interruptions (12): one per interrupted session
    for sid in sessions[~sessions.primary_analysis_eligible].session_id:
        plan.append(dict(session_id=sid, label="protocol_interruption", cluster="none"))

    # --- Background events up to 220
    bg_labels = ["bg_physio", "bg_movement", "bg_audio", "bg_light", "bg_single_sensor", "bg_env_audio"]
    bg_p = [0.22, 0.20, 0.18, 0.16, 0.14, 0.10]
    n_bg = 220 - len(plan)
    for _ in range(n_bg):
        site = rng.choice([s[0] for s in SITES])
        plan.append(dict(session_id=pick(site, 1)[0], label=rng.choice(bg_labels, p=bg_p), cluster="none"))
    return plan


def inject_events(rng, windows: pd.DataFrame, sessions: pd.DataFrame, plan: list[dict]):
    """Modify windows in place according to the label; return event metadata."""
    win = windows
    sess = sessions.set_index("session_id")
    used = {}  # session_id -> lista de (start, end) ocupados
    events = []

    def free_slot(sid, n_windows, lo=330, hi=None):
        dur = sess.loc[sid, "actual_duration_minutes"] * 60
        hi = hi or min(1440, dur - 60)
        for _ in range(200):
            st = int(rng.integers(lo // 10, hi // 10)) * 10
            en = st + n_windows * 10
            if all(en + 60 <= a or st - 60 >= b for a, b in used.get(sid, [])):
                used.setdefault(sid, []).append((st, en))
                return st, en
        raise RuntimeError(f"no slot in {sid}")

    def idx(sid, st, en):
        m = (win.session_id == sid) & (win.window_start_sec >= st) & (win.window_start_sec < en)
        return win.index[m]

    def bump(ix, hr=0.0, rmssd_pct=0.0, beta_pct=0.0, motion=None, transients=0, peak=0.0,
             light=0.0, env=0.0, muse_q=None, polar_q=None):
        if hr:
            win.loc[ix, "heart_rate_bpm"] += hr + rng.normal(0, 1.0, len(ix))
            win.loc[ix, "rr_mean_ms"] = 60000 / win.loc[ix, "heart_rate_bpm"]
        if rmssd_pct:
            win.loc[ix, "rmssd_ms"] *= (1 + rmssd_pct / 100) + rng.normal(0, 0.03, len(ix))
        if beta_pct:
            b = win.loc[ix, "muse_beta_rel"] * ((1 + beta_pct / 100) + rng.normal(0, 0.02, len(ix)))
            others = ["muse_delta_rel", "muse_theta_rel", "muse_alpha_rel", "muse_gamma_rel"]
            rest = win.loc[ix, others].sum(axis=1)
            win.loc[ix, "muse_beta_rel"] = b
            for c in others:
                win.loc[ix, c] = win.loc[ix, c] * (1 - b) / rest
        if motion is not None:
            win.loc[ix, "bodycam_motion_score"] = np.clip(motion + rng.normal(0, 0.015, len(ix)), 0.005, 0.95)
            win.loc[ix, "muse_accel_mean_g"] = 0.02 + 0.25 * win.loc[ix, "bodycam_motion_score"]
            win.loc[ix, "muse_accel_peak_g"] = win.loc[ix, "muse_accel_mean_g"] * 3.5
        if transients:
            first = ix[:1] if len(ix) else ix
            win.loc[first, "audio_transient_count"] += transients
            win.loc[ix, "audio_peak_dbfs"] += peak
        if light:
            win.loc[ix, "bodycam_light_change_score"] += light
        if env:
            win.loc[ix, "environmental_noise_db"] += env
        if muse_q is not None:
            win.loc[ix, "muse_signal_quality"] = muse_q + rng.normal(0, 0.03, len(ix))
            win.loc[ix, "artifact_flag"] = True
        if polar_q is not None:
            win.loc[ix, "polar_signal_quality"] = polar_q + rng.normal(0, 0.02, len(ix))

    for ev in plan:
        sid, label = ev["session_id"], ev["label"]
        meta = dict(session_id=sid, label=label, cluster=ev["cluster"])
        L = int(rng.integers(5, 9))  # 50-80 s

        if label == "tp_full":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            bump(ix, hr=rng.uniform(8, 16), rmssd_pct=-rng.uniform(22, 45), beta_pct=rng.uniform(12, 24),
                 motion=rng.uniform(0.04, 0.11), transients=int(rng.integers(1, 5)), peak=rng.uniform(6, 12))
            meta.update(st=st, en=en, event_type="multimodal_coincidence",
                        modalities="polar,muse,audio,bodycam", detection="sensor_rule+manual_review",
                        obs="Concurrent physiological and EEG change occurred near an audio transient. Body-camera movement remained low.",
                        expl_status="unresolved", known=np.nan, env_match="false",
                        note="No immediate environmental explanation was documented; compare across independent sessions.",
                        audio="positive")
        elif label == "tp_partial":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            which = rng.choice(["hr_beta", "hr_rmssd", "rmssd_beta"])
            bump(ix,
                 hr=rng.uniform(4.0, 6.5) if "hr" in which else rng.uniform(0, 2),
                 rmssd_pct=-rng.uniform(13, 20) if "rmssd" in which else -rng.uniform(0, 6),
                 beta_pct=rng.uniform(7, 11) if "beta" in which else rng.uniform(0, 4),
                 motion=rng.uniform(0.06, 0.14), transients=int(rng.integers(1, 3)), peak=rng.uniform(4, 8))
            meta.update(st=st, en=en, event_type="multimodal_coincidence",
                        modalities="polar,muse,audio", detection="sensor_rule+field_log",
                        obs="Concurrent physiological and EEG change occurred near an audio transient. Body-camera movement remained low.",
                        expl_status="unresolved", known=np.nan, env_match="unknown",
                        note="Low-context event; retain for comparison.", audio="weak_positive")
        elif label == "tp_no_audio":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            bump(ix, hr=rng.uniform(8, 14), rmssd_pct=-rng.uniform(20, 40), beta_pct=rng.uniform(10, 20),
                 motion=rng.uniform(0.04, 0.11))
            meta.update(st=st, en=en, event_type="physiological_shift",
                        modalities="polar,muse", detection="sensor_rule+manual_review",
                        obs="Concurrent physiological and EEG change without an audio transient. Body-camera movement remained low.",
                        expl_status="unresolved", known=np.nan, env_match="false",
                        note="No immediate environmental explanation was documented; compare across independent sessions.",
                        audio="control_nothing")
        elif label == "hn_movement":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            bump(ix, hr=rng.uniform(7, 13), rmssd_pct=-rng.uniform(18, 35), beta_pct=rng.uniform(9, 16),
                 motion=rng.uniform(0.30, 0.50), transients=int(rng.integers(1, 4)), peak=rng.uniform(4, 9))
            meta.update(st=st, en=en, event_type="physiological_shift",
                        modalities="polar,muse,bodycam,muse_accel", detection="automated_screen",
                        obs="Concurrent physiological and EEG change occurred near an audio transient.",
                        expl_status="probable", known="Movement was visible on body camera during the physiological change.",
                        env_match="true", note="Context suggests a conventional or single-modality explanation.",
                        audio="env_source")
        elif label == "hn_degraded_quality":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            bump(ix, hr=rng.uniform(5, 9), rmssd_pct=-rng.uniform(10, 22), beta_pct=rng.uniform(15, 30),
                 motion=rng.uniform(0.05, 0.12), transients=int(rng.integers(0, 2)), peak=3,
                 muse_q=rng.uniform(0.45, 0.65))
            meta.update(st=st, en=en, event_type="single_sensor_deviation",
                        modalities="muse,polar", detection="automated_screen",
                        obs="EEG beta increase during a short signal-quality reduction.",
                        expl_status="probable", known="Short signal-quality reduction may explain the isolated deviation.",
                        env_match="unknown", note="Context suggests a conventional or single-modality explanation.",
                        audio="maybe")
        elif label == "hn_spiritbox_radio":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            bump(ix, hr=rng.uniform(2, 4.5), rmssd_pct=-rng.uniform(4, 12), beta_pct=rng.uniform(3, 7),
                 motion=rng.uniform(0.05, 0.13), transients=int(rng.integers(2, 6)), peak=rng.uniform(8, 14))
            meta.update(st=st, en=en, event_type="audio_transient",
                        modalities="audio,polar", detection="sensor_rule+field_log",
                        obs="Voice-like fragment heard while the Spirit Box was sweeping; mild physiological response.",
                        expl_status="unresolved", known=np.nan, env_match="unknown",
                        note="Low-context event; retain for comparison.", audio="spiritbox_voice")
        elif label == "hn_env_confounded":
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            bump(ix, hr=rng.uniform(8, 14), rmssd_pct=-rng.uniform(20, 40), beta_pct=rng.uniform(10, 20),
                 motion=rng.uniform(0.05, 0.12), transients=int(rng.integers(2, 6)), peak=rng.uniform(6, 12),
                 env=rng.uniform(6, 12))
            src = sess.loc[sid, "known_electrical_sources"].split(",")[0].replace("_", " ")
            meta.update(st=st, en=en, event_type="multimodal_coincidence",
                        modalities="polar,muse,audio,environment", detection="sensor_rule+field_log",
                        obs="Concurrent physiological and EEG change occurred near an audio transient during a documented electrical interval.",
                        expl_status="probable", known=f"{src.capitalize()} activity was documented during the interval.",
                        env_match="true", note="Retain as an explainable comparison case for model evaluation.",
                        audio="mechanical")
        elif label == "hn_startle_lag":
            st, en = free_slot(sid, L + 1)
            ix_all = idx(sid, st, en)
            ix_pre, ix_phys = ix_all[:1], ix_all[1:]
            bump(ix_pre, transients=int(rng.integers(1, 3)), peak=rng.uniform(9, 15))
            bump(ix_phys, hr=rng.uniform(7, 12), rmssd_pct=-rng.uniform(18, 32), beta_pct=rng.uniform(9, 16),
                 motion=rng.uniform(0.05, 0.13))
            meta.update(st=st, en=en, event_type="multimodal_coincidence",
                        modalities="polar,muse,audio,bodycam", detection="sensor_rule+manual_review",
                        obs="Concurrent physiological and EEG change occurred near an audio transient. Body-camera movement remained low.",
                        expl_status="unresolved", known=np.nan, env_match="unknown",
                        note="Low-context event; retain for comparison.", audio="unknown_transient")
        elif label == "protocol_interruption":
            dur = sess.loc[sid, "actual_duration_minutes"] * 60
            st = max(310, dur - 50)
            st = int(st // 10) * 10
            en = st + 40
            ix = idx(sid, st, en)
            bump(ix, motion=rng.uniform(0.3, 0.5), transients=2, peak=6, env=rng.uniform(5, 10))
            meta.update(st=st, en=en, event_type="protocol_interruption",
                        modalities="field_log,environment", detection="field_log",
                        obs="Session interrupted by a documented protocol deviation.",
                        expl_status="confirmed", known=SITE_NOTES[sess.loc[sid, "site_id"]],
                        env_match="true",
                        note="Exclude this session from primary analysis but retain for audit and robustness testing.",
                        audio="none")
        else:  # background
            st, en = free_slot(sid, L)
            ix = idx(sid, st, en)
            if label == "bg_physio":
                bump(ix, hr=rng.uniform(3, 6), rmssd_pct=-rng.uniform(5, 14), motion=rng.uniform(0.08, 0.22))
                meta.update(event_type="physiological_shift", modalities="polar", detection="automated_screen",
                            obs="Brief heart-rate / HRV change detected without persistent EEG or audio convergence.",
                            expl_status="probable", known="Participant movement or posture change is a plausible explanation.",
                            env_match="unknown", note="Context suggests a conventional or single-modality explanation.", audio="control")
            elif label == "bg_movement":
                bump(ix, hr=rng.uniform(2, 6), motion=rng.uniform(0.35, 0.55), transients=int(rng.integers(0, 3)), peak=3)
                meta.update(event_type="movement_event", modalities="bodycam,muse_accel", detection="automated_screen",
                            obs="Short movement burst detected in body-camera and accelerometer data.",
                            expl_status="confirmed", known=rng.choice(["Investigator movement was visible on body camera.", "Footstep / investigator handling was visible on body camera.", "Equipment handling is the most likely source."]),
                            env_match="true", note="Context suggests a conventional or single-modality explanation.", audio="control")
            elif label == "bg_audio":
                bump(ix, transients=int(rng.integers(1, 4)), peak=rng.uniform(5, 11))
                meta.update(event_type="audio_transient", modalities="audio", detection="automated_screen",
                            obs="Short audio transient detected without a sustained multi-sensor response.",
                            expl_status=rng.choice(["probable", "unresolved"], p=[0.7, 0.3]),
                            known=rng.choice(["Distant exterior noise was present but exact source was not visually confirmed.", np.nan], p=[0.7, 0.3]),
                            env_match=rng.choice(["true", "unknown"], p=[0.6, 0.4]),
                            note="Context suggests a conventional or single-modality explanation.", audio="bg_audio")
            elif label == "bg_light":
                bump(ix, light=rng.uniform(0.2, 0.4))
                meta.update(event_type="light_change", modalities="bodycam", detection="automated_screen",
                            obs="Brief body-camera light-level change detected.",
                            expl_status="probable", known=rng.choice(["Flashlight movement was visible on body camera.", "Exterior light spill or flashlight movement is a plausible source."]),
                            env_match="true", note="Context suggests a conventional or single-modality explanation.", audio="control")
            elif label == "bg_single_sensor":
                bump(ix, beta_pct=rng.uniform(6, 12), muse_q=rng.uniform(0.7, 0.85))
                meta.update(event_type="single_sensor_deviation", modalities="muse", detection="automated_screen",
                            obs="Isolated sensor deviation detected without corroboration from other modalities.",
                            expl_status="probable", known="Sensor contact adjustment was documented.",
                            env_match="unknown", note="Context suggests a conventional or single-modality explanation.", audio="control")
            else:  # bg_env_audio
                bump(ix, transients=int(rng.integers(1, 3)), peak=rng.uniform(4, 8), env=rng.uniform(5, 9))
                meta.update(event_type="environmental_or_audio_event", modalities="audio,environment", detection="sensor_rule+field_log",
                            obs="Concurrent sensor change detected during a documented environmental interval.",
                            expl_status="probable", known=SITE_NOTES[sess.loc[sid, "site_id"]],
                            env_match="true", note="Retain as an explainable comparison case for model evaluation.", audio="mechanical")
            meta.update(st=st, en=en)
        events.append(meta)
    return events


# --------------------------------------------------------------------------
# 4. Event table derived from the windows
# --------------------------------------------------------------------------
def build_events(windows: pd.DataFrame, sessions: pd.DataFrame, metas: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sess = sessions.set_index("session_id")
    metas = sorted(metas, key=lambda m: (m["session_id"], m["st"]))
    rows, truth = [], []
    for i, m in enumerate(metas, start=1):
        eid = f"EVT_{i:04d}"
        w = windows[windows.session_id == m["session_id"]]
        dur = w[(w.window_start_sec >= m["st"]) & (w.window_start_sec < m["en"])]
        pre = w[(w.window_start_sec >= m["st"] - 60) & (w.window_start_sec < m["st"])]
        q = min(dur.polar_signal_quality.mean(), dur.muse_signal_quality.mean())
        quality = "good" if q >= 0.88 else ("acceptable" if q >= 0.75 else "degraded")
        rows.append(dict(
            event_id=eid, session_id=m["session_id"], site_id=sess.loc[m["session_id"], "site_id"],
            participant_id=sess.loc[m["session_id"], "participant_id"],
            event_start_sec=int(m["st"]), event_end_sec=int(m["en"]), phase="during",
            investigation_zone=zone_for(m["st"]), event_type=m["event_type"],
            source_modalities=m["modalities"], detection_method=m["detection"],
            observation_text=m["obs"], explanation_status=m["expl_status"],
            known_explanation=m["known"], environmental_match=m["env_match"],
            manual_marker=m["detection"] != "automated_screen", signal_quality_summary=quality,
            hr_delta_bpm_vs_pre_event=round(dur.heart_rate_bpm.mean() - pre.heart_rate_bpm.mean(), 1),
            rmssd_delta_pct_vs_pre_event=round(100 * (dur.rmssd_ms.mean() / pre.rmssd_ms.mean() - 1), 1),
            beta_delta_pct_vs_pre_event=round(100 * (dur.muse_beta_rel.mean() / pre.muse_beta_rel.mean() - 1), 1),
            audio_peak_dbfs=round(dur.audio_peak_dbfs.max(), 1),
            audio_transient_count=int(dur.audio_transient_count.sum()),
            bodycam_motion_score=round(dur.bodycam_motion_score.mean(), 3),
            environmental_noise_db=round(dur.environmental_noise_db.mean(), 1),
            review_note=m["note"],
        ))
        truth.append(dict(event_id=eid, session_id=m["session_id"], generator_label=m["label"],
                          is_designed_anomaly=m["label"].startswith("tp_"),
                          designed_cluster=m["cluster"], audio_design=m["audio"]))
    events = pd.DataFrame(rows)[EVENTS_COLS]
    return events, pd.DataFrame(truth)


# --------------------------------------------------------------------------
# 5. Independent audio review (coverage by protocol, not by event type)
# --------------------------------------------------------------------------
def build_audio_reviews(rng, events: pd.DataFrame, truth: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    sess = sessions.set_index("session_id")
    t = truth.set_index("event_id")
    rows = []
    n = 0
    for _, e in events.iterrows():
        design = t.loc[e.event_id, "audio_design"]
        has_audio = e.audio_transient_count >= 1 or "audio" in e.source_modalities
        # protocol: review every event with audio + a 20% random control sample
        if design == "none" or not (has_audio or rng.random() < 0.20):
            continue
        n += 1
        sb = rng.choice(["off", "on", "not_documented"], p=[0.50, 0.45, 0.05])
        rec = "partial" if sess.loc[e.session_id, "recording_sync_status"] == "partial" else "recording"
        if design == "positive":
            res = rng.choice(["voice_like", "short_transient", "ambiguous_noise"], p=[0.35, 0.45, 0.20])
            content = {"voice_like": rng.choice(["faint speech-like fragment; unintelligible", "brief vocal-like fragment; wording not reliable", "short low-level sound with speech-like cadence"]),
                       "short_transient": "brief isolated sound; source not identified",
                       "ambiguous_noise": "brief indistinct acoustic change"}[res]
            bleed = "unknown" if sb == "on" else "false"
            env_src = "false"
            conf = rng.uniform(0.55, 0.85)
            status = "unresolved"
            notes = "Independent recorder captured a sound with no documented environmental source."
        elif design == "weak_positive":
            res = rng.choice(["short_transient", "ambiguous_noise", "nothing_detected"], p=[0.5, 0.3, 0.2])
            content = "brief low-level sound" if res != "nothing_detected" else "no audible content in review window"
            bleed = "unknown" if sb == "on" else "false"
            env_src = rng.choice(["false", "unknown"])
            conf = rng.uniform(0.45, 0.70)
            status = "uncertain" if res != "nothing_detected" else "no_correspondence"
            notes = "Weak or ambiguous audio correspondence; reviewer confidence limited."
        elif design == "spiritbox_voice":
            sb = "on"
            res = "voice_like"
            content = rng.choice(["single transient with possible vocal quality", "brief vocal-like fragment; wording not reliable"])
            bleed = rng.choice(["possible", "unknown"], p=[0.7, 0.3])
            env_src = "unknown"
            conf = rng.uniform(0.40, 0.65)
            status = "uncertain"
            notes = "Voice-like fragment coincides with Spirit Box sweep; radio bleed cannot be excluded."
        elif design in ("mechanical", "env_source"):
            res = rng.choice(["mechanical_noise", "environmental_noise", "short_transient"], p=[0.5, 0.25, 0.25])
            content = rng.choice(["low mechanical hum and transient", "environmental noise matching field notes", "brief sound consistent with documented site activity", "mechanical cycling sound"])
            bleed = "false"
            env_src = "true"
            conf = rng.uniform(0.80, 0.97)
            status = "probable"
            notes = "Independent audio is consistent with the environmental explanation documented for the event."
        elif design == "unknown_transient":
            res = "short_transient"
            content = "brief isolated sound; source not identified"
            bleed = "unknown" if sb == "on" else "false"
            env_src = "unknown"
            conf = rng.uniform(0.55, 0.75)
            status = "uncertain"
            notes = "Short transient precedes the physiological change in the raw recording; source not identified."
        elif design == "bg_audio":
            res = rng.choice(["short_transient", "environmental_noise", "nothing_detected"], p=[0.5, 0.3, 0.2])
            content = rng.choice(["distant traffic or exterior noise", "short transient matching the field log", "brief low-level sound", "no audible content in review window"])
            bleed = "false"
            env_src = rng.choice(["true", "unknown"], p=[0.6, 0.4])
            conf = rng.uniform(0.6, 0.9)
            status = "probable" if env_src == "true" else "uncertain"
            notes = "Isolated transient without multi-sensor correspondence."
        elif design == "maybe":
            res = rng.choice(["nothing_detected", "short_transient"], p=[0.6, 0.4])
            content = "no audible content in review window" if res == "nothing_detected" else "brief low-level sound"
            bleed = "false"; env_src = "unknown"; conf = rng.uniform(0.6, 0.85)
            status = "no_correspondence" if res == "nothing_detected" else "uncertain"
            notes = "Audio review does not add evidence beyond the sensor deviation."
        else:  # control / control_nothing
            res = "nothing_detected"
            content = "no audible content in review window"
            bleed = "false"; env_src = "false"; conf = rng.uniform(0.85, 0.98)
            status = "no_correspondence"
            notes = "Control review: no audio correspondence with the sensor event."
        rows.append(dict(
            audio_review_id=f"AR_{n:04d}", event_id=e.event_id, session_id=e.session_id,
            site_id=e.site_id, participant_id=e.participant_id,
            review_start_sec=int(e.event_start_sec - 30), review_end_sec=int(e.event_end_sec + 30),
            recorder_status=rec, spirit_box_status=sb, audio_result=res, audible_content=content,
            speech_intelligibility="low" if res == "voice_like" else "none",
            radio_bleed_possible=bleed, environmental_source_possible=env_src,
            reviewer_confidence=round(conf, 2), explanation_status=status, review_notes=notes,
        ))
    return pd.DataFrame(rows)[AUDIO_COLS]


# --------------------------------------------------------------------------
# 6. Compatibility validation (same assertions as the notebook)
# --------------------------------------------------------------------------
ML_FEATURES = ["hr_delta_bpm_vs_pre_event", "rmssd_delta_pct_vs_pre_event", "beta_delta_pct_vs_pre_event",
               "audio_peak_dbfs", "audio_transient_count", "bodycam_motion_score", "environmental_noise_db"]


def validate(sessions, windows, events, audio):
    assert list(sessions.columns) == SESSIONS_COLS
    assert list(windows.columns) == WINDOWS_COLS
    assert list(events.columns) == EVENTS_COLS
    assert list(audio.columns) == AUDIO_COLS
    assert windows.session_id.isin(sessions.session_id).all()
    assert events.session_id.isin(sessions.session_id).all()
    assert audio.event_id.isin(events.event_id).all()
    assert events.event_id.is_unique and audio.event_id.is_unique
    assert (events.event_end_sec >= events.event_start_sec).all()
    assert events[ML_FEATURES].notna().all().all()
    assert len(events) == 220
    for must in ["SITE_14_P01_A"]:
        assert must in set(sessions.session_id), must
    for eid in ["EVT_0001", "EVT_0015", "EVT_0043"]:
        assert eid in set(events.event_id), eid
    np.testing.assert_allclose(windows[["muse_delta_rel", "muse_theta_rel", "muse_alpha_rel", "muse_beta_rel", "muse_gamma_rel"]].sum(axis=1), 1, atol=5e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paranormal_v2")
    ap.add_argument("--seed", type=int, default=333)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sessions = make_sessions(rng)
    windows = make_windows(rng, sessions)
    plan = event_plan(rng, sessions)
    metas = inject_events(rng, windows, sessions, plan)
    events, truth = build_events(windows, sessions, metas)
    audio = build_audio_reviews(rng, events, truth, sessions)

    # final rounding of window values
    eeg_cols = ["muse_delta_rel", "muse_theta_rel", "muse_alpha_rel", "muse_beta_rel", "muse_gamma_rel"]
    num = [c for c in windows.select_dtypes("float").columns if c not in eeg_cols]
    windows[num] = windows[num].round(4)
    windows[eeg_cols] = windows[eeg_cols].round(5)
    for c in ["heart_rate_bpm", "rr_mean_ms", "rmssd_ms", "sdnn_ms", "audio_rms_dbfs", "audio_peak_dbfs",
              "temperature_c", "humidity_pct", "environmental_noise_db"]:
        windows[c] = windows[c].round(1)
    windows = windows[WINDOWS_COLS]

    validate(sessions, windows, events, audio)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(out / "sessions.csv", index=False)
    windows.to_csv(out / "sensor_windows.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    audio.to_csv(out / "audio_reviews.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)

    print(f"seed={args.seed} -> {out.resolve()}")
    print("sessions", sessions.shape, "| windows", windows.shape, "| events", events.shape, "| audio_reviews", audio.shape)
    print("\nDesign labels:")
    print(truth.generator_label.value_counts().to_string())
    print("\naudio_reviews coverage by event type:")
    events["reviewed"] = events.event_id.isin(audio.event_id)
    print(pd.crosstab(events.event_type, events.reviewed).to_string())


if __name__ == "__main__":
    main()
