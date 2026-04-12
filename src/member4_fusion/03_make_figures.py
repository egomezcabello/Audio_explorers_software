#!/usr/bin/env python3
"""
03_make_figures.py – Generate report figures.

Creates publication-ready figures summarising the scene analysis results.

Planned figures:
  - DoA heatmap over time
  - Candidate score comparison (bar chart)
  - Waveform of the talker of interest
  - VAD timeline for each candidate

TODO:
    - Implement each figure generation function.
    - Use src.common.plotting helpers for consistent style.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, FIGURES_DIR, FINAL_DIR, ANALYSIS_DIR, ensure_output_dirs
from src.common.plotting import save_figure

logger = setup_logging(__name__)


def plot_doa_heatmap(save_path: Path) -> None:
    """DoA posterior heatmap with candidate track overlays."""
    posteriors_path = DOA_DIR / "doa_posteriors.npy"
    if not posteriors_path.exists():
        logger.warning("DoA posteriors not found – skipping heatmap.")
        return

    heatmap = np.load(str(posteriors_path))  # (n_grid, n_frames)
    n_grid, n_frames = heatmap.shape

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(heatmap, aspect="auto", origin="lower",
              extent=[0, n_frames, 0, 360], cmap="inferno",
              interpolation="bilinear")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Azimuth (°)")
    ax.set_title("DoA Posterior Heatmap with Tracks")

    # Overlay tracks from doa_tracks.json
    tracks_path = DOA_DIR / "doa_tracks.json"
    if tracks_path.exists():
        with open(tracks_path, "r", encoding="utf-8") as fh:
            scene = json.load(fh)
        colours = ["cyan", "lime", "yellow", "magenta",
                   "red", "orange", "white", "deepskyblue"]
        for i, cand in enumerate(scene.get("candidates", [])):
            cid = cand.get("id", f"spk{i:02d}")
            track = cand.get("doa_track", [])
            if track:
                frames = [e[0] for e in track]
                azimuths = [e[1] for e in track]
                ax.scatter(frames, azimuths, s=1,
                           c=colours[i % len(colours)],
                           alpha=0.5, label=cid)
        # Also show provisionals faintly
        for i, cand in enumerate(scene.get("provisional_candidates", [])):
            cid = cand.get("id", f"prov{i:02d}")
            track = cand.get("doa_track", [])
            if track:
                frames = [e[0] for e in track]
                azimuths = [e[1] for e in track]
                ax.scatter(frames, azimuths, s=0.5, c="gray",
                           alpha=0.3, label=cid)
        ax.legend(loc="upper right", fontsize=7, markerscale=5,
                  ncol=2, framealpha=0.7)
    save_figure(fig, save_path)
    logger.info("Figure saved → %s", save_path)


def plot_candidate_scores(save_path: Path) -> None:
    """Bar chart comparing candidate total scores with component breakdown."""
    scores_path = FINAL_DIR / "candidate_scores.json"
    if not scores_path.exists():
        logger.warning("Candidate scores not found – skipping bar chart.")
        return

    with open(scores_path, "r", encoding="utf-8") as fh:
        scores = json.load(fh)

    ids = [s["candidate_id"] for s in scores]
    totals = [s["total_score"] for s in scores]
    components = ["doa_stability", "speech_duration", "language_match", "snr_estimate"]
    comp_colours = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

    fig, ax = plt.subplots(figsize=(8, max(3, len(ids) * 0.5 + 1)))
    y_pos = np.arange(len(ids))
    left = np.zeros(len(ids))
    for comp, colour in zip(components, comp_colours):
        vals = np.array([s.get(comp, 0) for s in scores])
        ax.barh(y_pos, vals, left=left, color=colour, alpha=0.85,
                label=comp.replace("_", " ").title(), height=0.6)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ids)
    ax.set_xlabel("Score")
    ax.set_title("Candidate Score Breakdown")
    ax.legend(loc="lower right", fontsize=7)
    ax.invert_yaxis()
    save_figure(fig, save_path)
    logger.info("Figure saved → %s", save_path)


def plot_vad_timeline(save_path: Path) -> None:
    """VAD segments per candidate on a shared timeline."""
    analysis_files = sorted(ANALYSIS_DIR.glob("*_analysis.json"))
    if not analysis_files:
        logger.warning("No analysis files – skipping VAD timeline.")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(analysis_files) * 0.6)))
    colours = plt.cm.tab10.colors

    for i, af in enumerate(analysis_files):
        with open(af, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cid = data.get("candidate_id", af.stem.replace("_analysis", ""))
        segs = data.get("vad_segments", [])
        for seg in segs:
            if len(seg) >= 2:
                ax.barh(i, seg[1] - seg[0], left=seg[0], height=0.6,
                        color=colours[i % len(colours)], alpha=0.7)
        ax.text(-0.3, i, cid, ha="right", va="center", fontsize=9)

    ax.set_xlabel("Time (s)")
    ax.set_title("VAD Segments per Candidate")
    ax.set_yticks([])
    ax.invert_yaxis()
    save_figure(fig, save_path)
    logger.info("Figure saved → %s", save_path)


def plot_scene_summary_table(save_path: Path) -> None:
    """Render a summary table as a figure."""
    analysis_files = sorted(ANALYSIS_DIR.glob("*_analysis.json"))
    if not analysis_files:
        logger.warning("No analysis files – skipping summary table.")
        return

    # Load scores if available
    scores_data: Dict[str, Any] = {}
    scores_path = FINAL_DIR / "candidate_scores.json"
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as fh:
            for s in json.load(fh):
                scores_data[s["candidate_id"]] = s

    headers = ["ID", "F0 (Hz)", "Type", "Lang", "Score",
               "Transcript (first 60 chars)"]
    rows = []
    for af in analysis_files:
        with open(af, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cid = data.get("candidate_id", "?")
        pitch = data.get("pitch", {})
        f0 = pitch.get("median_f0_hz", 0)
        vtype = pitch.get("voice_type", "?")
        lang = data.get("language", "?")
        tx = data.get("transcript", "")[:60]
        sc = scores_data.get(cid, {}).get("total_score", 0)
        rows.append([cid, f"{f0:.0f}", vtype, lang, f"{sc:.2f}", tx])

    fig, ax = plt.subplots(figsize=(14, max(2, 0.5 * len(rows) + 1)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Scene Summary", fontsize=12, pad=10)
    save_figure(fig, save_path)
    logger.info("Figure saved → %s", save_path)


def main() -> None:
    """Entry point for step 03 (figures)."""
    ensure_output_dirs()

    logger.info("Generating report figures …")

    plot_doa_heatmap(FIGURES_DIR / "doa_heatmap.png")
    plot_candidate_scores(FIGURES_DIR / "candidate_scores.png")
    plot_vad_timeline(FIGURES_DIR / "vad_timeline.png")
    plot_scene_summary_table(FIGURES_DIR / "scene_summary_table.png")

    logger.info("Step 03 (figures) complete.")


if __name__ == "__main__":
    main()
