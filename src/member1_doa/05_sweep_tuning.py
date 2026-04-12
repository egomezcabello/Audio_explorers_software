#!/usr/bin/env python3
"""
05_sweep_tuning.py – Member 1 parameter sweep / tuning utility.
================================================================
Runs many Member 1 configurations, evaluates each against example
ground truth plus mixture sanity checks, and ranks results.

Usage
-----
    python -m src.member1_doa.05_sweep_tuning
    python -m src.member1_doa.05_sweep_tuning --max-runs 100
    python -m src.member1_doa.05_sweep_tuning --seed 123 --output-name my_sweep

The script does NOT permanently modify config.yaml.  It mutates the
in-memory ``CFG`` dict before each run and restores it afterwards.
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
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from src.common.config import CFG
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, OUTPUTS_DIR, ensure_output_dirs

logger = setup_logging(__name__)

# ── Tuning output directory ───────────────────────────────────────────
TUNING_DIR: Path = OUTPUTS_DIR / "doa_tuning"

# ── Example ground truth ──────────────────────────────────────────────
EXAMPLE_EXPECTED = [0.0, 90.0, 180.0, 270.0]
MATCH_THRESHOLD_DEG = 15.0


# ── Search space ──────────────────────────────────────────────────────

SEARCH_SPACE: Dict[str, List[Any]] = {
    # Calibration (step 01)
    "calibration_confidence_quantile":         [0.30, 0.40, 0.50, 0.60],
    "calibration_target_frames_per_direction": [100, 150, 200, 250, 300],
    # Hybrid scoring (step 02)
    "template_score_weight":                   [0.20, 0.30, 0.40, 0.50],
    "delay_mismatch_sigma_us":                 [15, 20, 25, 30, 35],
    # Discovery / peak detection (step 03)
    "min_peak_distance_deg":                   [20, 25, 30, 35, 40],
    "second_peak_ratio":                       [0.20, 0.30, 0.40, 0.50, 0.60],
    "global_peak_prominence":                  [0.05, 0.075, 0.10, 0.15],
    "max_assign_dist_deg":                     [15, 20, 25, 30],
    # Smart filtering — confirmation thresholds (step 03)
    "burst_score_ratio":                       [0.25, 0.30, 0.35, 0.40, 0.50],
    "min_group_count":                         [1, 2, 3],
    "group_threshold_ratio":                   [0.20, 0.25, 0.30, 0.35, 0.40],
    # Track quality (step 03)
    "min_track_points_frac":                   [0.02, 0.03, 0.04, 0.05],
    "min_track_duration_s":                    [0.3, 0.5, 0.8, 1.0],
}

PARAM_KEYS = list(SEARCH_SPACE.keys())


# ── Helpers ───────────────────────────────────────────────────────────

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

    Returns
    -------
    mean_err : mean nearest-neighbour error for each expected direction
    misses   : expected directions with no match within threshold
    false    : detected directions with no match within threshold
    """
    if not detected:
        return 180.0, len(expected), 0

    errs = []
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
    pks, props = find_peaks(tiled, distance=min_distance)
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


# ── Run one configuration ────────────────────────────────────────────

def _run_pipeline(overrides: Dict[str, Any]) -> None:
    """Run calibration + DoA + tracking for example and mixture."""
    import importlib

    # Force reimport to pick up CFG changes (modules read CFG at call time)
    mod_calib = importlib.import_module("src.member1_doa.01_calibrate_templates")
    mod_doa   = importlib.import_module("src.member1_doa.02_doa_estimate")
    mod_track = importlib.import_module("src.member1_doa.03_track_and_cluster")

    mod_calib.main(tag="example")
    mod_doa.main(tag="example")
    mod_track.main(tag="example")
    mod_doa.main(tag="mixture")
    mod_track.main(tag="mixture")


def _evaluate() -> Dict[str, Any]:
    """Read outputs and compute all metrics."""
    metrics: Dict[str, Any] = {}

    # ── Example tracks ────────────────────────────────────────────────
    ex_tracks_path = DOA_DIR / "example_doa_tracks.json"
    with open(ex_tracks_path, "r") as fh:
        ex = json.load(fh)
    ex_confirmed = ex.get("candidates", [])
    ex_provisional = ex.get("provisional_candidates", [])
    ex_azimuths = [c["mean_azimuth"] for c in ex_confirmed]
    metrics["example_confirmed_count"] = len(ex_confirmed)
    metrics["example_provisional_count"] = len(ex_provisional)
    # Legacy alias for backward compat
    metrics["example_track_count"] = len(ex_confirmed)

    mean_err, misses, false = _match_directions(EXAMPLE_EXPECTED, ex_azimuths)
    metrics["example_track_mean_err"] = round(mean_err, 2)
    metrics["example_track_misses"] = misses
    metrics["example_track_false"] = false

    # Over-finding diagnostic: count confirmed pairs within ~15° that
    # overlap heavily in time (IoU > 0.5).  This directly targets the
    # "one true speaker split into two confirmed tracks" failure.
    close_overlap_count = 0
    for i in range(len(ex_confirmed)):
        for j in range(i + 1, len(ex_confirmed)):
            sep = _angdist(ex_confirmed[i]["mean_azimuth"],
                           ex_confirmed[j]["mean_azimuth"])
            if sep > 15.0:
                continue
            # Compute temporal overlap from active_segments
            frames_i = set()
            for seg in ex_confirmed[i].get("active_segments", []):
                # Convert time bounds to a rough frame set
                for f in range(int(seg[0] * 100), int(seg[1] * 100) + 1):
                    frames_i.add(f)
            frames_j = set()
            for seg in ex_confirmed[j].get("active_segments", []):
                for f in range(int(seg[0] * 100), int(seg[1] * 100) + 1):
                    frames_j.add(f)
            inter = len(frames_i & frames_j)
            union = len(frames_i | frames_j)
            iou = inter / max(union, 1)
            if iou > 0.5:
                close_overlap_count += 1
    metrics["example_close_overlap_pairs"] = close_overlap_count

    # Strong provisional: provisional candidates with good burst evidence
    strong_prov = 0
    for p in ex_provisional:
        burst = p.get("mean_top3_window_score_hybrid", 0.0)
        if burst >= 0.4:
            strong_prov += 1
    metrics["example_strong_provisional"] = strong_prov

    # ── Example DoA posterior ─────────────────────────────────────────
    ex_post = np.load(str(DOA_DIR / "example_doa_posteriors.npy"))
    ex_avg = ex_post.mean(axis=1)
    ex_doa_peaks = _find_top_k_peaks(ex_avg, k=4)
    metrics["example_doa_count"] = len(ex_doa_peaks)

    doa_mean_err, doa_misses, doa_false = _match_directions(
        EXAMPLE_EXPECTED, ex_doa_peaks,
    )
    metrics["example_doa_mean_err"] = round(doa_mean_err, 2)
    metrics["example_doa_misses"] = doa_misses
    metrics["example_doa_false"] = doa_false

    # ── Mixture tracks ────────────────────────────────────────────────
    mix_tracks_path = DOA_DIR / "mixture_doa_tracks.json"
    with open(mix_tracks_path, "r") as fh:
        mix = json.load(fh)
    mix_confirmed = mix.get("candidates", [])
    mix_provisional = mix.get("provisional_candidates", [])
    metrics["mixture_confirmed_count"] = len(mix_confirmed)
    metrics["mixture_provisional_count"] = len(mix_provisional)
    # Legacy alias
    metrics["mixture_track_count"] = len(mix_confirmed)
    if mix_confirmed:
        metrics["mixture_mean_track_score"] = round(
            float(np.mean([c.get("mean_score_hybrid",
                                 c.get("mean_score", 0))
                           for c in mix_confirmed])), 4)
        metrics["mixture_mean_track_duration"] = round(
            float(np.mean([c["total_duration_s"] for c in mix_confirmed])), 2)
    else:
        metrics["mixture_mean_track_score"] = 0.0
        metrics["mixture_mean_track_duration"] = 0.0

    # ── Mixture posterior peak count ──────────────────────────────────
    mix_post = np.load(str(DOA_DIR / "mixture_doa_posteriors.npy"))
    mix_avg = mix_post.mean(axis=1)
    mix_peaks = _find_top_k_peaks(mix_avg, k=12, min_distance=25)
    metrics["mixture_peak_count"] = len(mix_peaks)
    metrics["mixture_total_tracks"] = (
        metrics["mixture_confirmed_count"]
        + metrics["mixture_provisional_count"]
    )

    return metrics


def _apply_hard_rejection(m: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (rejected, reason).  Hard rules:
    - example must have 3-5 confirmed tracks (relaxed from exactly 4
      to let the sweep explore borderline configurations)
    - no close-overlap pairs (two confirmed within 15° sharing time)
    - mixture must have >= 5 total tracks (confirmed + provisional)
    """
    ex_n = m.get("example_confirmed_count", m.get("example_track_count", 0))
    if ex_n < 3 or ex_n > 5:
        return True, f"example_confirmed={ex_n} (need 3-5)"
    if m.get("example_close_overlap_pairs", 0) > 0:
        return True, f"example has close-overlap pairs (over-finding)"
    if m["example_track_mean_err"] > 20.0:
        return True, f"example_mean_err={m['example_track_mean_err']:.1f}° (>20)"
    if m["example_track_misses"] > 1:
        return True, f"example_misses={m['example_track_misses']} (need <=1)"
    mix_total = m.get("mixture_total_tracks", 0)
    if mix_total < 5:
        return True, f"mixture_total={mix_total} (need >=5)"
    return False, ""


def _compute_score(m: Dict[str, Any], rejected: bool) -> float:
    """Compute the final scalar score (lower is better)."""
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

    # Over-finding penalty: confirmed tracks within ~15° that share time
    close_pairs = m.get("example_close_overlap_pairs", 0)
    score += 25.0 * close_pairs

    # Under-finding reward: strong provisionals indicate missed real speakers
    strong_prov = m.get("example_strong_provisional", 0)
    if strong_prov > 0:
        score += 5.0 * strong_prov  # penalise leaving strong candidates unconfirmed

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


# ── Sample configurations ────────────────────────────────────────────

def _sample_random(n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Sample n random configs from the search space."""
    configs = []
    for _ in range(n):
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        # Convert numpy types to plain Python
        cfg = {k: (int(v) if isinstance(v, (np.integer,))
                    else float(v) if isinstance(v, (np.floating,))
                    else v)
               for k, v in cfg.items()}
        configs.append(cfg)
    return configs


def _build_grid() -> List[Dict[str, Any]]:
    """Build the full grid (warning: can be huge)."""
    keys = list(SEARCH_SPACE.keys())
    vals = [SEARCH_SPACE[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ── CSV columns ───────────────────────────────────────────────────────

CSV_COLUMNS = [
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
    "example_close_overlap_pairs",
    "example_strong_provisional",
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


# ── Main sweep ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Member 1 DoA parameter sweep / tuning",
    )
    parser.add_argument("--max-runs", type=int, default=50,
                        help="Max number of configs to test (default: 50)")
    parser.add_argument("--random-search", action="store_true", default=True,
                        help="Use random search (default)")
    parser.add_argument("--grid", action="store_true", default=False,
                        help="Use full grid instead of random search")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-name", type=str, default="member1_sweep",
                        help="Base name for output files")
    args = parser.parse_args()

    ensure_output_dirs()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    # Suppress verbose logging from pipeline steps during sweep
    for mod_name in [
        "src.member1_doa.01_calibrate_templates",
        "src.member1_doa.02_doa_estimate",
        "src.member1_doa.03_track_and_cluster",
    ]:
        logging.getLogger(mod_name).setLevel(logging.WARNING)

    rng = np.random.default_rng(args.seed)

    # ── Build config list ─────────────────────────────────────────────
    if args.grid:
        configs = _build_grid()
        logger.info("Grid mode: %d total configs", len(configs))
        if len(configs) > args.max_runs:
            rng.shuffle(configs)
            configs = configs[:args.max_runs]
            logger.info("Capped to %d runs (--max-runs)", args.max_runs)
    else:
        configs = _sample_random(args.max_runs, rng)
        logger.info("Random search: %d configs (seed=%d)",
                    len(configs), args.seed)

    # ── Save original CFG['doa'] ──────────────────────────────────────
    original_doa = copy.deepcopy(CFG.get("doa", {}))

    # ── CSV output ────────────────────────────────────────────────────
    csv_path = TUNING_DIR / f"{args.output_name}_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, config in enumerate(configs):
        run_id = i + 1
        short = " ".join(f"{k[:8]}={v}" for k, v in config.items())
        logger.info("[sweep] run %d/%d: %s", run_id, len(configs), short)

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

            logger.info("[sweep] run %d: score=%.1f  ex_err=%.1f  "
                        "ex_n=%d  mix_n=%d  %s",
                        run_id, row["total_score"],
                        metrics["example_track_mean_err"],
                        metrics["example_track_count"],
                        metrics["mixture_track_count"],
                        "REJECTED" if rejected else "ok")

        except Exception as exc:
            logger.warning("[sweep] run %d FAILED: %s", run_id, exc)
            traceback.print_exc()
            row["rejected"] = True
            row["total_score"] = 9999.0
            row["error"] = str(exc)[:200]
            # Fill missing metrics with defaults
            for col in CSV_COLUMNS:
                if col not in row:
                    row[col] = ""

        writer.writerow(row)
        csv_file.flush()
        all_rows.append(row)

    csv_file.close()

    # ── Restore original config ───────────────────────────────────────
    CFG["doa"] = original_doa

    elapsed = time.time() - t0
    logger.info("[sweep] %d runs completed in %.0f s (%.1f s/run)",
                len(all_rows), elapsed, elapsed / max(len(all_rows), 1))

    # ── Rank results ──────────────────────────────────────────────────
    valid = [r for r in all_rows if r.get("total_score", 9999) < 1000]
    valid.sort(key=lambda r: r["total_score"])
    rejected_count = sum(1 for r in all_rows
                         if r.get("rejected") or r.get("total_score", 0) >= 1000)

    # ── Terminal summary ──────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 65)
    logger.info("Total runs    : %d", len(all_rows))
    logger.info("Valid runs    : %d", len(valid))
    logger.info("Rejected runs : %d", rejected_count)

    if valid:
        scores = [r["total_score"] for r in valid]
        logger.info("Score range   : %.1f – %.1f", min(scores), max(scores))
        logger.info("Median score  : %.1f", float(np.median(scores)))
        logger.info("-" * 65)
        logger.info("TOP 5 CONFIGS:")
        for rank, r in enumerate(valid[:5], 1):
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
        logger.warning("No valid runs! All were rejected or failed.")

    # ── JSON summary ──────────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "total_runs": len(all_rows),
        "valid_runs": len(valid),
        "rejected_runs": rejected_count,
        "elapsed_s": round(elapsed, 1),
        "score_range": ([min(scores), max(scores)] if valid else []),
        "median_score": round(float(np.median(scores)), 2) if valid else None,
        "best_config": valid[0] if valid else None,
        "top5": valid[:5] if valid else [],
    }
    json_path = TUNING_DIR / f"{args.output_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("[sweep] CSV  -> %s", csv_path)
    logger.info("[sweep] JSON -> %s", json_path)

    # ── Optional plots ────────────────────────────────────────────────
    _try_plot(all_rows, args.output_name)

    logger.info("[sweep] done.")


# ── Simple plots (best-effort) ────────────────────────────────────────

def _try_plot(rows: List[Dict[str, Any]], name: str) -> None:
    """Generate simple diagnostic plots. Silently skip on import error."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.debug("[sweep] matplotlib not available — skipping plots")
        return

    valid = [r for r in rows
             if isinstance(r.get("total_score"), (int, float))
             and r["total_score"] < 1000]
    if len(valid) < 2:
        return

    scores = [r["total_score"] for r in valid]

    # 1. Histogram of scores
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Total score (lower = better)")
    ax.set_ylabel("Count")
    ax.set_title("Member 1 sweep – score distribution")
    fig.tight_layout()
    fig.savefig(str(TUNING_DIR / f"{name}_hist.png"), dpi=120)
    plt.close(fig)

    # 2. Scatter: example_track_mean_err vs mixture_total_tracks
    ex_errs = [r.get("example_track_mean_err", 0) for r in valid]
    mix_n = [r.get("mixture_total_tracks",
                   r.get("mixture_track_count", 0)) for r in valid]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(mix_n, ex_errs, c=scores, cmap="viridis_r",
                    edgecolors="k", linewidths=0.5, alpha=0.8)
    ax.set_xlabel("mixture_total_tracks (confirmed + provisional)")
    ax.set_ylabel("example_track_mean_err (°)")
    ax.set_title("Example error vs mixture total tracks")
    fig.colorbar(sc, ax=ax, label="total_score")
    fig.tight_layout()
    fig.savefig(str(TUNING_DIR / f"{name}_scatter.png"), dpi=120)
    plt.close(fig)

    logger.info("[sweep] plots saved to %s", TUNING_DIR)


if __name__ == "__main__":
    main()
