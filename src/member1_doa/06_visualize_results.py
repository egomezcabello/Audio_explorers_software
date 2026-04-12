#!/usr/bin/env python3
"""
06_visualize_results.py – Visualize Member 1 DoA results.
==========================================================
Final step of the Member 1 (DoA) pipeline.

Generates four visualizations per tag:

1. **DoA heatmap** with track overlays
2. **Time-averaged angular spectrum** with detected peaks marked
3. **Polar plot** of final speaker directions
4. **Expected-vs-detected comparison** (example tag only)

Also prints concise summaries to the logger.

Usage
-----
    python -m src.member1_doa.06_visualize_results --tag example
    python -m src.member1_doa.06_visualize_results --tag mixture
    python -m src.member1_doa.06_visualize_results --tag all
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.config import CFG, get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, ensure_output_dirs
from src.common.plotting import save_figure

logger = setup_logging(__name__)

EXAMPLE_EXPECTED_DIRS = [0.0, 90.0, 180.0, 270.0]


# ── Data loaders ──────────────────────────────────────────────────────

def load_posteriors(tag: str) -> Optional[np.ndarray]:
    """Load DoA posteriors for *tag*.  Returns None if missing."""
    path = DOA_DIR / f"{tag}_doa_posteriors.npy"
    if not path.exists():
        logger.warning("[viz] posteriors not found: %s", path.name)
        return None
    return np.load(str(path))


def load_tracks(tag: str) -> Optional[dict]:
    """Load DoA tracks JSON for *tag*.  Returns None if missing."""
    path = DOA_DIR / f"{tag}_doa_tracks.json"
    if not path.exists():
        logger.warning("[viz] tracks not found: %s", path.name)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Helpers ───────────────────────────────────────────────────────────

def _angular_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _frame_to_seconds(n_frames: int) -> np.ndarray:
    hop = get_stft_params().get("hop_length", 256)
    return np.arange(n_frames) * hop / SAMPLE_RATE


def match_expected_to_detected(
    expected: List[float],
    detected: List[float],
) -> List[Tuple[float, float, float]]:
    """
    Greedily match expected directions to nearest detected directions.

    Returns list of ``(expected, detected, error)`` tuples.
    """
    used = set()
    matches = []
    for exp in expected:
        best_d, best_det = 999.0, exp
        for i, det in enumerate(detected):
            if i in used:
                continue
            d = _angular_distance(exp, det)
            if d < best_d:
                best_d, best_det = d, det
                best_i = i
        if best_d < 999.0:
            used.add(best_i)
        matches.append((exp, best_det, best_d))
    return matches


def summarize_tracks(tag: str, scene: dict) -> None:
    """Print a compact track summary to the logger."""
    candidates = scene.get("candidates", [])
    if not candidates:
        logger.info("[%s][viz] no tracks", tag)
        return
    az = [f"{c['mean_azimuth']:.1f}" for c in candidates]
    dur = [f"{c['total_duration_s']:.1f}" for c in candidates]
    sc = [f"{c['mean_score']:.2f}" for c in candidates]
    logger.info("[%s][viz] %d tracks: az=[%s] dur=[%s] score=[%s]",
                tag, len(candidates),
                ",".join(az), ",".join(dur), ",".join(sc))

    if tag == "example":
        detected = [c["mean_azimuth"] for c in candidates]
        matches = match_expected_to_detected(EXAMPLE_EXPECTED_DIRS, detected)
        errs = [m[2] for m in matches]
        logger.info("[%s][viz] expected=%s  matched=%s",
                    tag,
                    [f"{m[0]:.0f}" for m in matches],
                    [f"{m[1]:.1f}" for m in matches])
        logger.info("[%s][viz] errors=%s  mean=%.1f°",
                    tag,
                    [f"{e:.1f}" for e in errs],
                    float(np.mean(errs)))


# ── Plot functions ────────────────────────────────────────────────────

def plot_heatmap_with_tracks(
    posteriors: np.ndarray,
    scene: Optional[dict],
    tag: str,
) -> None:
    """
    1) DoA heatmap over time with track overlays.

    Saves to ``outputs/doa/{tag}_plot_heatmap_tracks.png``.
    """
    n_grid, n_frames = posteriors.shape
    t_sec = _frame_to_seconds(n_frames)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(
        t_sec, np.arange(n_grid), posteriors,
        shading="auto", cmap="inferno", rasterized=True,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Azimuth (°)")
    ax.set_title(f"DoA posterior — {tag}")
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 45))
    fig.colorbar(im, ax=ax, label="Posterior strength", pad=0.02)

    # Overlay tracks
    if scene is not None:
        colors = plt.cm.tab10.colors
        dt = t_sec[1] if len(t_sec) > 1 else 1.0
        # Confirmed tracks (solid dots)
        for i, cand in enumerate(scene.get("candidates", [])):
            az_track = cand.get("azimuth_track", [])
            if not az_track:
                continue
            arr = np.array(az_track, dtype=float)
            valid = ~np.isnan(arr) if hasattr(arr, '__len__') else np.ones(len(arr), bool)
            frames = np.arange(len(arr))
            t_track = frames * dt
            c = colors[i % len(colors)]
            ax.plot(t_track[valid], arr[valid], '.', color=c,
                    markersize=1.0, alpha=0.7,
                    label=f"{cand.get('id', '')} ({cand['mean_azimuth']:.0f}°)")
        # Provisional tracks (open circles, dimmer)
        for j, prov in enumerate(scene.get("provisional_candidates", [])):
            az_track = prov.get("azimuth_track", [])
            if not az_track:
                continue
            arr = np.array(az_track, dtype=float)
            valid = ~np.isnan(arr) if hasattr(arr, '__len__') else np.ones(len(arr), bool)
            frames = np.arange(len(arr))
            t_track = frames * dt
            c = colors[(len(scene.get('candidates', [])) + j) % len(colors)]
            ax.plot(t_track[valid], arr[valid], '.', color=c,
                    markersize=0.6, alpha=0.35,
                    label=f"prov {prov.get('id', '')} ({prov['mean_azimuth']:.0f}°)")
        ax.legend(fontsize=7, loc="upper right", ncol=2,
                  markerscale=5, framealpha=0.8)

    save_path = DOA_DIR / f"{tag}_plot_heatmap_tracks.png"
    save_figure(fig, save_path)
    logger.info("[%s][viz] saved %s", tag, save_path.name)


def plot_avg_spectrum(
    posteriors: np.ndarray,
    scene: Optional[dict],
    tag: str,
) -> None:
    """
    2) Time-averaged angular spectrum with detected peaks marked.

    Saves to ``outputs/doa/{tag}_plot_avg_spectrum.png``.
    """
    n_grid = posteriors.shape[0]
    avg = posteriors.mean(axis=1)
    azimuths = np.arange(n_grid)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(azimuths, avg, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Mean posterior strength")
    ax.set_title(f"Time-averaged DoA spectrum — {tag}")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))

    # Mark detected track azimuths
    if scene is not None:
        for cand in scene.get("candidates", []):
            az = cand["mean_azimuth"]
            ax.axvline(az, color="red", alpha=0.6, linewidth=1.0, linestyle="--")
            ax.text(az, avg.max() * 0.95, f" {cand.get('id', '')}",
                    fontsize=8, color="red", rotation=90,
                    va="top", ha="left")
        for prov in scene.get("provisional_candidates", []):
            az = prov["mean_azimuth"]
            ax.axvline(az, color="gray", alpha=0.5, linewidth=0.8, linestyle=":")
            ax.text(az, avg.max() * 0.85, f" p{prov.get('id', '')}",
                    fontsize=7, color="gray", rotation=90,
                    va="top", ha="left")

    # Log dominant peaks
    from scipy.signal import find_peaks as _find_peaks
    peaks, props = _find_peaks(avg, distance=30, height=avg.max() * 0.1)
    if len(peaks) > 0:
        pk_str = [f"{p}°" for p in sorted(peaks)]
        logger.info("[%s][viz] avg-spectrum peaks: %s", tag, pk_str)

    save_path = DOA_DIR / f"{tag}_plot_avg_spectrum.png"
    save_figure(fig, save_path)
    logger.info("[%s][viz] saved %s", tag, save_path.name)


def plot_peak_count_histogram(
    posteriors: np.ndarray,
    scene: Optional[dict],
    tag: str,
    min_distance_deg: int = 30,
) -> None:
    """
    Peak-count histogram: how often each angle was a top-1 or top-2
    peak across frames.

    This diagnostic directly corresponds to intermittent/late speakers
    that the time-averaged spectrum hides: a direction that is rarely
    the strongest but frequently the second-strongest will show up
    here but not in the blue-line average.

    Saves to ``outputs/doa/{tag}_plot_peak_histogram.png``.
    """
    from scipy.signal import find_peaks as _find_peaks

    n_grid, n_frames = posteriors.shape
    top1_hist = np.zeros(n_grid, dtype=np.float64)
    top2_hist = np.zeros(n_grid, dtype=np.float64)

    for t in range(n_frames):
        spectrum = posteriors[:, t]
        fmax = spectrum.max()
        if fmax < 1e-8:
            continue

        pad = min_distance_deg + 2
        tiled = np.concatenate([spectrum[-pad:], spectrum, spectrum[:pad]])
        pks, props = _find_peaks(tiled, distance=min_distance_deg,
                                 height=0.1 * fmax)
        pks_orig = pks - pad
        valid = (pks_orig >= 0) & (pks_orig < n_grid)
        pks_orig = pks_orig[valid]
        heights = props["peak_heights"][valid]

        if len(pks_orig) == 0:
            continue

        order = np.argsort(heights)[::-1]
        # Top-1 peak
        top1_hist[pks_orig[order[0]]] += 1
        # Top-2 peak (if exists)
        if len(order) >= 2:
            top2_hist[pks_orig[order[1]]] += 1

    # Smooth slightly for readability
    from scipy.ndimage import gaussian_filter1d as _gf1d
    top1_smooth = _gf1d(top1_hist, sigma=2, mode='wrap')
    top2_smooth = _gf1d(top2_hist, sigma=2, mode='wrap')

    azimuths = np.arange(n_grid)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(azimuths, top1_smooth, alpha=0.4, color="steelblue",
                    label="Top-1 peak count")
    ax.fill_between(azimuths, top2_smooth, alpha=0.4, color="coral",
                    label="Top-2 peak count")
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Frame count")
    ax.set_title(f"Peak-count histogram over time — {tag}")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(fontsize=9)

    # Mark detected tracks
    if scene is not None:
        for cand in scene.get("candidates", []):
            az = cand["mean_azimuth"]
            ax.axvline(az, color="red", alpha=0.5, linewidth=1.0,
                       linestyle="--")
        for prov in scene.get("provisional_candidates", []):
            az = prov["mean_azimuth"]
            ax.axvline(az, color="gray", alpha=0.4, linewidth=0.8,
                       linestyle=":")

    save_path = DOA_DIR / f"{tag}_plot_peak_histogram.png"
    save_figure(fig, save_path)
    logger.info("[%s][viz] saved %s", tag, save_path.name)


def plot_window_max_spectrum(
    posteriors: np.ndarray,
    scene: Optional[dict],
    tag: str,
    window_dur_s: float = 1.5,
) -> None:
    """
    Window-max angular spectrum: for each angle, the maximum of
    windowed time-averages.

    Unlike the global mean spectrum (blue line), this preserves
    evidence for late-starting or intermittent speakers whose best
    window is strong but whose global average is diluted.

    Saves to ``outputs/doa/{tag}_plot_window_max_spectrum.png``.
    """
    n_grid, n_frames = posteriors.shape
    hop = get_stft_params().get("hop_length", 256)
    win_frames = max(1, int(window_dur_s * SAMPLE_RATE / hop))
    step = max(1, win_frames // 2)

    window_max = np.zeros(n_grid, dtype=np.float64)
    t = 0
    while t + win_frames <= n_frames:
        win_avg = posteriors[:, t:t + win_frames].mean(axis=1)
        window_max = np.maximum(window_max, win_avg)
        t += step

    # Also compute the global mean for comparison
    global_avg = posteriors.mean(axis=1)

    azimuths = np.arange(n_grid)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(azimuths, global_avg, linewidth=1.2, color="steelblue",
            alpha=0.6, label="Global mean (blue line)")
    ax.plot(azimuths, window_max, linewidth=1.4, color="darkorange",
            label=f"Window-max ({window_dur_s:.1f}s)")
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Posterior strength")
    ax.set_title(f"Window-max vs global-mean spectrum — {tag}")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(fontsize=9)

    # Mark detected tracks
    if scene is not None:
        for cand in scene.get("candidates", []):
            az = cand["mean_azimuth"]
            ax.axvline(az, color="red", alpha=0.5, linewidth=1.0,
                       linestyle="--")
        for prov in scene.get("provisional_candidates", []):
            az = prov["mean_azimuth"]
            ax.axvline(az, color="gray", alpha=0.4, linewidth=0.8,
                       linestyle=":")

    save_path = DOA_DIR / f"{tag}_plot_window_max_spectrum.png"
    save_figure(fig, save_path)
    logger.info("[%s][viz] saved %s", tag, save_path.name)


def plot_polar_tracks(
    scene: dict,
    tag: str,
) -> None:
    """
    3) Polar plot of final speaker directions.

    Saves to ``outputs/doa/{tag}_plot_polar_tracks.png``.
    """
    candidates = scene.get("candidates", [])
    if not candidates:
        logger.info("[%s][viz] no candidates for polar plot", tag)
        return

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    colors = plt.cm.tab10.colors
    # Confirmed (solid, inner ring)
    for i, cand in enumerate(candidates):
        az_rad = np.deg2rad(cand["mean_azimuth"])
        c = colors[i % len(colors)]
        ax.plot(az_rad, 1.0, 'o', color=c, markersize=12)
        ax.annotate(
            f"  {cand.get('id', '')} ({cand['mean_azimuth']:.0f}°)",
            xy=(az_rad, 1.0), fontsize=8, color=c,
        )
    # Provisional (hollow, outer ring)
    provisionals = scene.get("provisional_candidates", [])
    for j, prov in enumerate(provisionals):
        az_rad = np.deg2rad(prov["mean_azimuth"])
        c = colors[(len(candidates) + j) % len(colors)]
        ax.plot(az_rad, 1.15, 'o', color=c, markersize=9,
                markerfacecolor='none', markeredgewidth=1.5)
        ax.annotate(
            f"  prov ({prov['mean_azimuth']:.0f}°)",
            xy=(az_rad, 1.15), fontsize=7, color=c, alpha=0.7,
        )

    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.set_title(f"Speaker directions — {tag}  ({len(candidates)}C + {len(provisionals)}P)", pad=20)

    save_path = DOA_DIR / f"{tag}_plot_polar_tracks.png"
    save_figure(fig, save_path)
    logger.info("[%s][viz] saved %s", tag, save_path.name)


def plot_example_expected_vs_detected(scene: dict) -> None:
    """
    4) Expected-vs-detected comparison (example tag only).

    Saves to ``outputs/doa/example_plot_expected_vs_detected.png``.
    """
    candidates = scene.get("candidates", [])
    detected = [c["mean_azimuth"] for c in candidates]
    if not detected:
        logger.info("[example][viz] no tracks — skipping comparison plot")
        return

    matches = match_expected_to_detected(EXAMPLE_EXPECTED_DIRS, detected)
    expected = [m[0] for m in matches]
    matched = [m[1] for m in matches]
    errors = [m[2] for m in matches]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: expected vs detected scatter
    ax1.plot([0, 360], [0, 360], 'k--', alpha=0.3, label="Perfect")
    ax1.scatter(expected, matched, s=80, c="steelblue", zorder=3)
    for e, d in zip(expected, matched):
        ax1.annotate(f"  {e:.0f}°→{d:.0f}°", (e, d), fontsize=8)
    ax1.set_xlabel("Expected azimuth (°)")
    ax1.set_ylabel("Detected azimuth (°)")
    ax1.set_title("Expected vs Detected")
    ax1.set_xlim(-10, 370)
    ax1.set_ylim(-10, 370)
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8)

    # Right: angular error bar chart
    labels = [f"{e:.0f}°" for e in expected]
    ax2.bar(labels, errors, color="salmon", edgecolor="darkred", width=0.5)
    ax2.axhline(float(np.mean(errors)), color="red", linestyle="--",
                alpha=0.6, label=f"mean={np.mean(errors):.1f}°")
    ax2.set_xlabel("Expected direction")
    ax2.set_ylabel("Angular error (°)")
    ax2.set_title("Per-direction error")
    ax2.legend(fontsize=8)

    fig.suptitle("Example — expected vs detected directions", fontsize=13)
    fig.tight_layout()

    save_path = DOA_DIR / "example_plot_expected_vs_detected.png"
    save_figure(fig, save_path)
    logger.info("[example][viz] saved %s", save_path.name)


# ── Entry point ────────────────────────────────────────────────────────

def main(tag: str = "mixture") -> None:
    """
    Generate visualizations for the given tag.

    Parameters
    ----------
    tag : str
        ``"example"``, ``"mixture"``, or ``"all"`` (both).
    """
    ensure_output_dirs()

    if tag == "all":
        for t in ["example", "mixture"]:
            main(t)
        return

    logger.info("[%s][viz] generating plots ...", tag)

    posteriors = load_posteriors(tag)
    scene = load_tracks(tag)

    # Print summary
    if scene is not None:
        summarize_tracks(tag, scene)

    # 1. Heatmap
    if posteriors is not None:
        plot_heatmap_with_tracks(posteriors, scene, tag)
    else:
        logger.warning("[%s][viz] skipping heatmap — no posteriors", tag)

    # 2. Average spectrum
    if posteriors is not None:
        plot_avg_spectrum(posteriors, scene, tag)
    else:
        logger.warning("[%s][viz] skipping avg spectrum — no posteriors", tag)

    # 2b. Peak-count histogram (intermittent speaker diagnostic)
    if posteriors is not None:
        plot_peak_count_histogram(posteriors, scene, tag)

    # 2c. Window-max spectrum (late/bursty speaker diagnostic)
    if posteriors is not None:
        plot_window_max_spectrum(posteriors, scene, tag)

    # 3. Polar plot
    if scene is not None:
        plot_polar_tracks(scene, tag)
    else:
        logger.warning("[%s][viz] skipping polar — no tracks", tag)

    # 4. Expected vs detected (example only)
    if tag == "example" and scene is not None:
        plot_example_expected_vs_detected(scene)

    logger.info("[%s][viz] done", tag)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(
        description="Step 04 – Visualize Member 1 DoA results")
    _p.add_argument("--tag", default="all",
                    help="Tag to visualize: example, mixture, or all (default: all)")
    main(_p.parse_args().tag)
