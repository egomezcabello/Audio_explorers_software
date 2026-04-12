#!/usr/bin/env python3
"""
05_smaller_sweep.py – Focused local parameter sweep around the best M1 config.
================================================================================
Samples configurations from a *narrow* neighbourhood around a known-good
parameter set and evaluates each against example ground truth + mixture
sanity, exactly like the broad sweep (04_sweep_tuning.py).

Differences from the broad sweep
---------------------------------
* **Smaller, centred search space** – each parameter varies only ±1-2
  steps around the current best, not across the full range.
* **Self-contained** – does not import or modify 04_sweep_tuning.py.
* **Separate outputs** – results go to ``outputs/member1_sweeps/smaller_sweep/``
  so old broad-sweep outputs are never touched.
* **Optional best-config export** – ``--write-best-config`` dumps a
  YAML snippet with the winning parameter values.

Usage
-----
    python -m src.member1_doa.05_smaller_sweep
    python -m src.member1_doa.05_smaller_sweep --runs 80 --seed 99
    python -m src.member1_doa.05_smaller_sweep --write-best-config
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from src.common.config import CFG
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, OUTPUTS_DIR, ensure_output_dirs

logger = setup_logging(__name__)

# ── Output directory (separate from the broad sweep) ──────────────────
DEFAULT_OUTDIR: Path = OUTPUTS_DIR / "member1_sweeps" / "smaller_sweep"

# ── Example ground truth ──────────────────────────────────────────────
EXAMPLE_EXPECTED: List[float] = [0.0, 90.0, 180.0, 270.0]
MATCH_THRESHOLD_DEG: float = 15.0

# ── Centre point: dynamically loaded from the broad sweep best config ─
_BROAD_SWEEP_SUMMARY = OUTPUTS_DIR / "doa_tuning" / "member1_sweep_summary.json"
_CENTRE_KEYS = [
    "calibration_confidence_quantile",
    "calibration_target_frames_per_direction",
    "template_score_weight",
    "delay_mismatch_sigma_us",
    "min_peak_distance_deg",
    "second_peak_ratio",
    "global_peak_prominence",
    "max_assign_dist_deg",
    "burst_score_ratio",
    "min_group_count",
    "group_threshold_ratio",
    "min_track_points_frac",
    "min_track_duration_s",
]

# Defaults from config.yaml — used when broad sweep summary is unavailable
_CONFIG_DEFAULTS: Dict[str, Any] = {
    "calibration_confidence_quantile": 0.40,
    "calibration_target_frames_per_direction": 200,
    "template_score_weight": 0.40,
    "delay_mismatch_sigma_us": 25.0,
    "min_peak_distance_deg": 25,
    "second_peak_ratio": 0.30,
    "global_peak_prominence": 0.10,
    "max_assign_dist_deg": 20.0,
    "burst_score_ratio": 0.40,
    "min_group_count": 2,
    "group_threshold_ratio": 0.30,
    "min_track_points_frac": 0.03,
    "min_track_duration_s": 0.5,
}

def _load_centre() -> Dict[str, Any]:
    """Read best_config from the broad sweep summary JSON.
    Falls back to config.yaml defaults if no sweep summary exists."""
    if _BROAD_SWEEP_SUMMARY.exists():
        with open(_BROAD_SWEEP_SUMMARY) as f:
            summary = json.load(f)
        best = summary["best_config"]
        centre = {k: best[k] for k in _CENTRE_KEYS if k in best}
        # Fill any missing keys from defaults
        for k in _CENTRE_KEYS:
            if k not in centre:
                centre[k] = _CONFIG_DEFAULTS[k]
        score = best.get("total_score", 0)
        logger.info("[local] centre loaded from %s  (score=%.1f)",
                    _BROAD_SWEEP_SUMMARY.name, score)
    else:
        logger.warning("[local] no broad sweep summary found; using config.yaml defaults")
        centre = {k: _CONFIG_DEFAULTS[k] for k in _CENTRE_KEYS}
    return centre


# Per-parameter definition: (step_size, n_steps_each_side, min_val, max_val)
# Used by _make_local_space() to build a tight neighbourhood.
PARAM_DEFS: Dict[str, Tuple] = {
    "calibration_confidence_quantile":         (0.05,  2, 0.10, 0.90),
    "calibration_target_frames_per_direction": (25,    2, 50,   500),
    "template_score_weight":                   (0.05,  2, 0.05, 0.95),
    "delay_mismatch_sigma_us":                 (5,     2, 5,    50),
    "min_peak_distance_deg":                   (5,     2, 15,   60),
    "second_peak_ratio":                       (0.05,  2, 0.10, 0.80),
    "global_peak_prominence":                  (0.025, 2, 0.025, 0.30),
    "max_assign_dist_deg":                     (5,     2, 5,    40),
    "burst_score_ratio":                       (0.05,  2, 0.10, 0.80),
    "min_group_count":                         (1,     1, 1,    4),
    "group_threshold_ratio":                   (0.05,  2, 0.10, 0.70),
    "min_track_points_frac":                   (0.01,  2, 0.01, 0.15),
    "min_track_duration_s":                    (0.25,  2, 0.25, 3.0),
}


def _make_local_space(centre: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Build a tight search space around *centre* using PARAM_DEFS."""
    space: Dict[str, List[Any]] = {}
    for key in _CENTRE_KEYS:
        c = centre[key]
        step, n, lo, hi = PARAM_DEFS[key]
        is_int = isinstance(c, int) and isinstance(step, int)
        vals = set()
        for offset in range(-n, n + 1):
            v = c + offset * step
            if v < lo or v > hi:
                continue
            if is_int:
                v = int(v)
            else:
                v = round(float(v), 4)
            vals.add(v)
        space[key] = sorted(vals)
    return space


CENTRE: Dict[str, Any] = _load_centre()
LOCAL_SPACE: Dict[str, List[Any]] = _make_local_space(CENTRE)
PARAM_KEYS: List[str] = list(LOCAL_SPACE.keys())


# ═══════════════════════════════════════════════════════════════════════
#  Helpers (self-contained, intentionally duplicated from 05)
# ═══════════════════════════════════════════════════════════════════════

def _angdist(a: float, b: float) -> float:
    """Shortest angular distance in degrees (0–180)."""
    d = abs(a - b) % 360
    return d if d <= 180.0 else 360.0 - d


def _match_directions(
    expected: List[float],
    detected: List[float],
    threshold: float = MATCH_THRESHOLD_DEG,
) -> Tuple[float, int, int]:
    """
    Match expected to detected directions.

    Returns (mean_err, misses, false_count).
    """
    if not detected:
        return 180.0, len(expected), 0

    errs: List[float] = []
    misses = 0
    for e in expected:
        dists = [_angdist(e, d) for d in detected]
        best = min(dists)
        errs.append(best)
        if best > threshold:
            misses += 1

    false_count = 0
    for d in detected:
        dists = [_angdist(d, e) for e in expected]
        if min(dists) > threshold:
            false_count += 1

    mean_err = float(np.mean(errs)) if errs else 180.0
    return mean_err, misses, false_count


def _find_top_k_peaks(
    spectrum: np.ndarray,
    k: int = 4,
    min_distance: int = 30,
) -> List[float]:
    """Find top-k peaks in a 1-D circular spectrum."""
    pad = min_distance + 2
    tiled = np.concatenate([spectrum[-pad:], spectrum, spectrum[:pad]])
    pks, _ = find_peaks(tiled, distance=min_distance)
    pks_orig = pks - pad
    valid = (pks_orig >= 0) & (pks_orig < len(spectrum))
    pks_orig = pks_orig[valid]
    heights = tiled[pks[valid]]
    order = np.argsort(heights)[::-1][:k]
    return sorted(float(pks_orig[i]) for i in order)


# ── Config override context manager ──────────────────────────────────

class _ConfigOverride:
    """Temporarily override keys in CFG['doa'] and restore on exit."""

    def __init__(self, overrides: Dict[str, Any]):
        self._overrides = overrides
        self._saved: Dict[str, Any] = {}

    def __enter__(self):
        doa = CFG.setdefault("doa", {})
        for key, val in self._overrides.items():
            self._saved[key] = doa.get(key)
            doa[key] = val
        return self

    def __exit__(self, *exc):
        doa = CFG["doa"]
        for key, old_val in self._saved.items():
            if old_val is None:
                doa.pop(key, None)
            else:
                doa[key] = old_val


# ── Pipeline runner ───────────────────────────────────────────────────

def _run_pipeline(overrides: Dict[str, Any]) -> None:
    """Run calibration → DoA → tracking for example + mixture."""
    import importlib

    mod_calib = importlib.import_module("src.member1_doa.01_calibrate_templates")
    mod_doa   = importlib.import_module("src.member1_doa.02_doa_estimate")
    mod_track = importlib.import_module("src.member1_doa.03_track_and_cluster")

    mod_calib.main(tag="example")
    mod_doa.main(tag="example")
    mod_track.main(tag="example")
    mod_doa.main(tag="mixture")
    mod_track.main(tag="mixture")


# ── Evaluation ────────────────────────────────────────────────────────

def _evaluate() -> Dict[str, Any]:
    """Read pipeline outputs and compute all metrics."""
    m: Dict[str, Any] = {}

    # ── Example tracks ────────────────────────────────────────────────
    with open(DOA_DIR / "example_doa_tracks.json", "r") as fh:
        ex = json.load(fh)
    ex_confirmed = ex.get("candidates", [])
    ex_provisional = ex.get("provisional_candidates", [])
    ex_az = [c["mean_azimuth"] for c in ex_confirmed]
    m["example_confirmed_count"] = len(ex_confirmed)
    m["example_provisional_count"] = len(ex_provisional)
    m["example_track_count"] = len(ex_confirmed)

    mean_err, misses, false = _match_directions(EXAMPLE_EXPECTED, ex_az)
    m["example_track_mean_err"] = round(mean_err, 2)
    m["example_track_misses"] = misses
    m["example_track_false"] = false

    # ── Example DoA posterior ─────────────────────────────────────────
    ex_post = np.load(str(DOA_DIR / "example_doa_posteriors.npy"))
    ex_avg = ex_post.mean(axis=1)
    ex_doa_peaks = _find_top_k_peaks(ex_avg, k=4)
    m["example_doa_count"] = len(ex_doa_peaks)

    doa_mean_err, doa_misses, doa_false = _match_directions(
        EXAMPLE_EXPECTED, ex_doa_peaks)
    m["example_doa_mean_err"] = round(doa_mean_err, 2)
    m["example_doa_misses"] = doa_misses
    m["example_doa_false"] = doa_false

    # ── Mixture tracks ────────────────────────────────────────────────
    with open(DOA_DIR / "mixture_doa_tracks.json", "r") as fh:
        mix = json.load(fh)
    mix_confirmed = mix.get("candidates", [])
    mix_provisional = mix.get("provisional_candidates", [])
    m["mixture_confirmed_count"] = len(mix_confirmed)
    m["mixture_provisional_count"] = len(mix_provisional)
    m["mixture_track_count"] = len(mix_confirmed)
    if mix_confirmed:
        m["mixture_mean_track_score"] = round(
            float(np.mean([c.get("mean_score_hybrid",
                                 c.get("mean_score", 0))
                           for c in mix_confirmed])), 4)
        m["mixture_mean_track_duration"] = round(
            float(np.mean([c["total_duration_s"] for c in mix_confirmed])), 2)
    else:
        m["mixture_mean_track_score"] = 0.0
        m["mixture_mean_track_duration"] = 0.0

    # ── Mixture posterior peak count ──────────────────────────────────
    mix_post = np.load(str(DOA_DIR / "mixture_doa_posteriors.npy"))
    mix_avg = mix_post.mean(axis=1)
    mix_peaks = _find_top_k_peaks(mix_avg, k=12, min_distance=25)
    m["mixture_peak_count"] = len(mix_peaks)
    m["mixture_total_tracks"] = (
        m["mixture_confirmed_count"]
        + m["mixture_provisional_count"]
    )

    return m


# ── Hard rejection ────────────────────────────────────────────────────

def _apply_hard_rejection(m: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Return (rejected, reason).  Hard rules:
    - example must have exactly 4 confirmed tracks
    - mixture must have >= 7 total tracks (confirmed + provisional)
    """
    ex_n = m.get("example_confirmed_count", m.get("example_track_count", 0))
    if ex_n != 4:
        return True, f"example_confirmed={ex_n} (need exactly 4)"
    if m["example_track_mean_err"] > 20.0:
        return True, f"example_mean_err={m['example_track_mean_err']:.1f}° (>20)"
    if m["example_track_misses"] > 0:
        return True, f"example_misses={m['example_track_misses']} (need 0)"
    mix_total = m.get("mixture_total_tracks", 0)
    if mix_total < 7:
        return True, f"mixture_total={mix_total} (need >=7)"
    return False, ""


# ── Scoring (same philosophy as 05, lower = better) ───────────────────

def _compute_score(m: Dict[str, Any], rejected: bool) -> float:
    """Weighted sum of error metrics.  Lower is better."""
    ex_n = m.get("example_confirmed_count", m.get("example_track_count", 0))
    mix_n = m.get("mixture_confirmed_count", m.get("mixture_track_count", 0))
    mix_total = m.get("mixture_total_tracks", 0)
    ex_prov = m.get("example_provisional_count", 0)

    score = (
        4.0 * m["example_track_mean_err"]
        + 2.0 * m["example_doa_mean_err"]
        + 20.0 * abs(ex_n - 4)
        + 15.0 * m["example_track_misses"]
        + 10.0 * m["example_track_false"]
        + 3.0 * abs(m["example_doa_count"] - 4)
        + 2.0 * m["example_doa_misses"]
    )

    # Provisional clutter in example (mild)
    if ex_prov > 6:
        score += 1.0 * (ex_prov - 6)

    # Mixture: reward more total tracks, penalise falling below 7
    if mix_total < 7:
        score += 8.0 * (7 - mix_total)

    # Mixture quality
    if mix_n < 3:
        score += 5.0 * (3 - mix_n)
    if m["mixture_mean_track_score"] < 0.50:
        score += 3.0

    if rejected:
        score += 1000.0

    return round(score, 2)


# ── Sampling ──────────────────────────────────────────────────────────

def _sample_local(n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """
    Sample n configs from LOCAL_SPACE.  The centre config is always
    included as run #1 so we have a baseline.
    """
    configs: List[Dict[str, Any]] = []

    # Always include the centre point first
    configs.append({k: v for k, v in CENTRE.items()})

    for _ in range(n - 1):
        cfg: Dict[str, Any] = {}
        for k, vals in LOCAL_SPACE.items():
            v = rng.choice(vals)
            # Normalise numpy scalars to plain Python types
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            cfg[k] = v
        configs.append(cfg)

    return configs


# ── CSV columns ───────────────────────────────────────────────────────

CSV_COLUMNS: List[str] = [
    "run_id",
    *PARAM_KEYS,
    "example_confirmed_count",
    "example_provisional_count",
    "example_track_count",
    "example_track_mean_err",
    "example_track_misses",
    "example_track_false",
    "example_doa_count",
    "example_doa_mean_err",
    "example_doa_misses",
    "example_doa_false",
    "mixture_confirmed_count",
    "mixture_provisional_count",
    "mixture_total_tracks",
    "mixture_track_count",
    "mixture_mean_track_score",
    "mixture_mean_track_duration",
    "mixture_peak_count",
    "rejected",
    "rejection_reason",
    "total_score",
    "error",
]


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Member 1 – focused local parameter sweep",
    )
    parser.add_argument("--runs", type=int, default=50,
                        help="Number of configs to evaluate (default: 50)")
    parser.add_argument("--seed", type=int, default=77,
                        help="Random seed for reproducibility (default: 77)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: outputs/member1_sweeps/smaller_sweep)")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Optional user tags for bookkeeping (stored in summary)")
    parser.add_argument("--write-best-config", action="store_true", default=False,
                        help="Write the best config as a YAML snippet")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs()

    logger.info("=" * 65)
    logger.info("Member 1 – focused local sweep (%d runs, seed=%d)",
                args.runs, args.seed)
    logger.info("Output dir: %s", outdir)
    logger.info("=" * 65)

    # Suppress verbose pipeline logging during sweep
    for mod_name in [
        "src.member1_doa.01_calibrate_templates",
        "src.member1_doa.02_doa_estimate",
        "src.member1_doa.03_track_and_cluster",
    ]:
        logging.getLogger(mod_name).setLevel(logging.WARNING)

    rng = np.random.default_rng(args.seed)
    configs = _sample_local(args.runs, rng)

    # Snapshot original config so we can restore even on Ctrl-C
    original_doa = copy.deepcopy(CFG.get("doa", {}))

    # ── CSV writer ────────────────────────────────────────────────────
    csv_path = outdir / "smaller_sweep_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    try:
        for i, config in enumerate(configs):
            run_id = i + 1
            short = " ".join(f"{k[:7]}={v}" for k, v in config.items())
            logger.info("[local] run %d/%d: %s", run_id, len(configs), short)

            row: Dict[str, Any] = {"run_id": run_id, **config}

            try:
                with _ConfigOverride(config):
                    _run_pipeline(config)
                    metrics = _evaluate()

                row.update(metrics)
                rejected, reason = _apply_hard_rejection(metrics)
                row["rejected"] = rejected
                row["rejection_reason"] = reason
                row["total_score"] = _compute_score(metrics, rejected)
                row["error"] = ""

                tag = "REJECTED" if rejected else "ok"
                logger.info(
                    "[local] run %d: score=%6.1f  ex_err=%4.1f  "
                    "ex_n=%d  mix_n=%d  %s",
                    run_id, row["total_score"],
                    metrics["example_track_mean_err"],
                    metrics["example_track_count"],
                    metrics["mixture_track_count"],
                    tag,
                )

            except Exception as exc:
                logger.warning("[local] run %d FAILED: %s", run_id, exc)
                traceback.print_exc()
                row["rejected"] = True
                row["rejection_reason"] = "exception"
                row["total_score"] = 9999.0
                row["error"] = str(exc)[:200]
                for col in CSV_COLUMNS:
                    if col not in row:
                        row[col] = ""

            writer.writerow(row)
            csv_file.flush()
            all_rows.append(row)

    finally:
        # Always restore original config, even on KeyboardInterrupt
        CFG["doa"] = original_doa
        csv_file.close()
        logger.info("[local] original config restored.")

    elapsed = time.time() - t0
    logger.info("[local] %d runs in %.0f s (%.1f s/run)",
                len(all_rows), elapsed, elapsed / max(len(all_rows), 1))

    # ── Rank results ──────────────────────────────────────────────────
    valid = [r for r in all_rows
             if isinstance(r.get("total_score"), (int, float))
             and r["total_score"] < 1000]
    valid.sort(key=lambda r: r["total_score"])
    rejected_rows = [r for r in all_rows
                     if r.get("rejected") or r.get("total_score", 0) >= 1000]

    # ── Terminal summary ──────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("LOCAL SWEEP SUMMARY")
    logger.info("=" * 65)
    logger.info("Total runs    : %d", len(all_rows))
    logger.info("Valid runs    : %d", len(valid))
    logger.info("Rejected runs : %d", len(rejected_rows))

    if valid:
        scores = [r["total_score"] for r in valid]
        logger.info("Score range   : %.1f – %.1f", min(scores), max(scores))
        logger.info("Median score  : %.1f", float(np.median(scores)))
        logger.info("-" * 65)
        logger.info("TOP 10 CONFIGS:")
        for rank, r in enumerate(valid[:10], 1):
            logger.info(
                "  #%d  score=%5.1f  ex_err=%4.1f  ex_n=%d  "
                "mix=%d+%d=%d  | bsr=%.2f  mgc=%s  gtr=%.2f  "
                "mpd=%s  spr=%.2f  gpp=%.3f  mad=%s",
                rank, r["total_score"],
                r.get("example_track_mean_err", 0),
                r.get("example_confirmed_count",
                      r.get("example_track_count", 0)),
                r.get("mixture_confirmed_count", 0),
                r.get("mixture_provisional_count", 0),
                r.get("mixture_total_tracks", 0),
                r.get("burst_score_ratio", 0),
                r.get("min_group_count", "?"),
                r.get("group_threshold_ratio", 0),
                r.get("min_peak_distance_deg", "?"),
                r.get("second_peak_ratio", 0),
                r.get("global_peak_prominence", 0),
                r.get("max_assign_dist_deg", "?"),
            )
    else:
        logger.warning("No valid runs – all were rejected or failed.")

    # ── JSON: full summary ────────────────────────────────────────────
    summary = {
        "sweep_type": "local_smaller_sweep",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "centre_config": CENTRE,
        "seed": args.seed,
        "total_runs": len(all_rows),
        "valid_runs": len(valid),
        "rejected_runs": len(rejected_rows),
        "elapsed_s": round(elapsed, 1),
        "score_range": [min(scores), max(scores)] if valid else [],
        "median_score": round(float(np.median(scores)), 2) if valid else None,
        "best_config": valid[0] if valid else None,
        "top5": valid[:5],
        "tags": args.tags or [],
    }
    json_path = outdir / "smaller_sweep_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("[local] summary -> %s", json_path)

    # ── JSON: top 10 ──────────────────────────────────────────────────
    top10_path = outdir / "smaller_sweep_top10.json"
    with open(top10_path, "w", encoding="utf-8") as fh:
        json.dump(valid[:10], fh, indent=2, default=str)
    logger.info("[local] top10   -> %s", top10_path)

    # ── JSON: rejected runs ───────────────────────────────────────────
    rej_path = outdir / "smaller_sweep_rejected.json"
    with open(rej_path, "w", encoding="utf-8") as fh:
        json.dump(rejected_rows, fh, indent=2, default=str)
    logger.info("[local] rejected -> %s", rej_path)

    # ── Optional: best config as YAML snippet ─────────────────────────
    if args.write_best_config and valid:
        _write_best_yaml(valid[0], outdir)

    # ── Plots ─────────────────────────────────────────────────────────
    _try_plot(all_rows, outdir)

    logger.info("[local] done.")


# ── YAML export ───────────────────────────────────────────────────────

def _write_best_yaml(best: Dict[str, Any], outdir: Path) -> None:
    """Write the best config as a YAML snippet for easy copy-paste."""
    yaml_path = outdir / "smaller_sweep_best_config.yaml"
    lines = [
        "# Best config from smaller sweep",
        f"# score={best.get('total_score', '?')}  "
        f"ex_err={best.get('example_track_mean_err', '?')}  "
        f"ex_n={best.get('example_track_count', '?')}  "
        f"mix_n={best.get('mixture_track_count', '?')}",
        "doa:",
    ]
    for k in PARAM_KEYS:
        v = best.get(k, "?")
        lines.append(f"  {k}: {v}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("[local] best YAML -> %s", yaml_path)


# ── Diagnostic plots (best-effort) ───────────────────────────────────

def _try_plot(rows: List[Dict[str, Any]], outdir: Path) -> None:
    """Generate simple diagnostic plots.  Silently skip if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.debug("[local] matplotlib not available – skipping plots")
        return

    valid = [r for r in rows
             if isinstance(r.get("total_score"), (int, float))
             and r["total_score"] < 1000]
    if len(valid) < 2:
        return

    scores = [r["total_score"] for r in valid]

    # 1. Histogram of valid scores
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, edgecolor="black", alpha=0.75, color="#4c8bbe")
    ax.set_xlabel("Total score (lower = better)")
    ax.set_ylabel("Count")
    ax.set_title("Local sweep – score distribution")
    fig.tight_layout()
    fig.savefig(str(outdir / "smaller_sweep_hist.png"), dpi=120)
    plt.close(fig)

    # 2. Scatter: example error vs mixture total tracks (colour = score)
    ex_errs = [r.get("example_track_mean_err", 0) for r in valid]
    mix_n   = [r.get("mixture_total_tracks",
                     r.get("mixture_track_count", 0)) for r in valid]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(mix_n, ex_errs, c=scores, cmap="viridis_r",
                    edgecolors="k", linewidths=0.5, alpha=0.8, s=50)
    ax.set_xlabel("mixture_total_tracks (confirmed + provisional)")
    ax.set_ylabel("example_track_mean_err (°)")
    ax.set_title("Local sweep – example error vs mixture total tracks")
    fig.colorbar(sc, ax=ax, label="total_score")
    fig.tight_layout()
    fig.savefig(str(outdir / "smaller_sweep_scatter.png"), dpi=120)
    plt.close(fig)

    # 3. Per-parameter box plots (score vs parameter value)
    for param in PARAM_KEYS:
        vals_set = sorted(set(r.get(param) for r in valid if r.get(param) is not None))
        if len(vals_set) < 2:
            continue
        groups = []
        labels = []
        for v in vals_set:
            g = [r["total_score"] for r in valid if r.get(param) == v]
            if g:
                groups.append(g)
                labels.append(str(v))
        if len(groups) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(groups, tick_labels=labels)
        ax.set_xlabel(param)
        ax.set_ylabel("total_score")
        ax.set_title(f"Score vs {param}")
        fig.tight_layout()
        fig.savefig(str(outdir / f"smaller_sweep_box_{param}.png"), dpi=100)
        plt.close(fig)

    logger.info("[local] plots saved to %s", outdir)


if __name__ == "__main__":
    main()
