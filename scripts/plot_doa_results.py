#!/usr/bin/env python3
"""
plot_doa_results.py – Visualise the Member 1 DoA pipeline outputs.

Produces three figures:
  1. DoA heatmap (time × azimuth posterior)
  2. Detected speaker tracks overlaid on the heatmap
  3. Calibration bar chart (TDOA per mic pair)

Usage
-----
    python scripts/plot_doa_results.py              # interactive display
    python scripts/plot_doa_results.py --save       # save PNGs to outputs/report/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Resolve project root (two levels up from scripts/) ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.paths import CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, FIGURES_DIR


def plot_heatmap(heatmap: np.ndarray, hop: int, sr: int,
                 title: str = "DoA Posterior Heatmap") -> plt.Figure:
    """Time × azimuth heatmap."""
    n_frames, n_grid = heatmap.shape
    duration = n_frames * hop / sr

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, 360],
        cmap="inferno",
        interpolation="bilinear",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Azimuth (°)")
    ax.set_title(title)
    ax.set_yticks(np.arange(0, 361, 45))
    fig.colorbar(im, ax=ax, label="Pseudo-probability", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_tracks(heatmap: np.ndarray, tracks_json: dict,
                hop: int, sr: int) -> plt.Figure:
    """Overlay detected speaker tracks on the heatmap."""
    n_frames, n_grid = heatmap.shape
    duration = n_frames * hop / sr

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, 360],
        cmap="inferno",
        interpolation="bilinear",
        alpha=0.7,
    )

    colours = plt.cm.tab10.colors
    candidates = tracks_json.get("candidates", [])
    for i, cand in enumerate(candidates):
        track = np.array(cand["doa_track"])  # (N, 2): [frame, az]
        if len(track) == 0:
            continue
        t = track[:, 0] * hop / sr
        az = track[:, 1]
        colour = colours[i % len(colours)]
        ax.scatter(t, az, s=1.5, color=colour, alpha=0.6, label=cand["id"])

        # Mark mean azimuth
        mean_az = float(np.mean(az))
        ax.axhline(mean_az, color=colour, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Azimuth (°)")
    ax.set_title("Detected Speaker Tracks")
    ax.set_yticks(np.arange(0, 361, 45))
    ax.legend(loc="upper right", fontsize=8, markerscale=5, ncol=2)
    fig.tight_layout()
    return fig


def plot_calibration(calib: dict) -> plt.Figure:
    """Bar chart of TDOA and confidence per mic pair."""
    pairs = calib.get("pairs", [])
    labels = [f"{p['mic_pair'][0]}–{p['mic_pair'][1]}" for p in pairs]
    tdoas = [p["tdoa_samples"] for p in pairs]
    confs = [p["confidence"] for p in pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(labels))
    ax1.bar(x, tdoas, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30)
    ax1.set_ylabel("TDOA (samples)")
    ax1.set_title("GCC-PHAT TDOA per Mic Pair")
    ax1.axhline(0, color="k", linewidth=0.5)

    ax2.bar(x, confs, color="darkorange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30)
    ax2.set_ylabel("Peak Confidence")
    ax2.set_title("GCC-PHAT Confidence per Mic Pair")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise DoA results")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to outputs/report/figures/ instead of displaying")
    args = parser.parse_args()

    stft_params = get_stft_params()
    hop = stft_params["hop_length"]
    sr = SAMPLE_RATE

    # Load outputs
    heatmap_path = DOA_DIR / "doa_posteriors.npy"
    tracks_path = DOA_DIR / "doa_tracks.json"
    calib_path = CALIB_DIR / "calibration.json"

    if not heatmap_path.exists():
        print(f"ERROR: Heatmap not found at {heatmap_path}. Run the pipeline first.")
        sys.exit(1)

    heatmap = np.load(str(heatmap_path))
    print(f"Loaded heatmap: {heatmap.shape}")

    tracks = {}
    if tracks_path.exists():
        with open(tracks_path) as f:
            tracks = json.load(f)
        n_cands = len(tracks.get("candidates", []))
        print(f"Loaded {n_cands} candidate track(s)")

    calib = {}
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        print(f"Loaded calibration ({len(calib.get('pairs', []))} pairs)")

    # Create figures
    fig1 = plot_heatmap(heatmap, hop, sr)
    fig2 = plot_tracks(heatmap, tracks, hop, sr) if tracks else None
    fig3 = plot_calibration(calib) if calib else None

    if args.save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig1.savefig(str(FIGURES_DIR / "doa_heatmap.png"), dpi=150, bbox_inches="tight")
        print(f"Saved → {FIGURES_DIR / 'doa_heatmap.png'}")
        if fig2:
            fig2.savefig(str(FIGURES_DIR / "doa_tracks.png"), dpi=150, bbox_inches="tight")
            print(f"Saved → {FIGURES_DIR / 'doa_tracks.png'}")
        if fig3:
            fig3.savefig(str(FIGURES_DIR / "calibration.png"), dpi=150, bbox_inches="tight")
            print(f"Saved → {FIGURES_DIR / 'calibration.png'}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
