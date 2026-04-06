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
from src.common.paths import DOA_DIR, FIGURES_DIR, FINAL_DIR, ensure_output_dirs
from src.common.plotting import save_figure

logger = setup_logging(__name__)


def plot_doa_heatmap(save_path: Path) -> None:
    """
    Plot the DoA azimuth-vs-time heatmap.

    TODO: Load doa_posteriors.npy and create a proper imshow plot.
    """
    posteriors_path = DOA_DIR / "doa_posteriors.npy"
    if posteriors_path.exists():
        heatmap = np.load(str(posteriors_path))
    else:
        logger.warning("DoA posteriors not found – using dummy for figure.")
        heatmap = np.random.rand(100, 360).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(heatmap.T, aspect="auto", origin="lower",
              extent=[0, heatmap.shape[0], 0, 360])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Azimuth (°)")
    ax.set_title("DoA Posterior Heatmap")
    save_figure(fig, save_path)
    logger.info("Figure saved → %s", save_path)


def plot_candidate_scores(save_path: Path) -> None:
    """
    Bar chart comparing candidate total scores.

    TODO: Load candidate_scores.json and plot.
    """
    scores_path = FINAL_DIR / "candidate_scores.json"
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as fh:
            scores = json.load(fh)
    else:
        logger.warning("Candidate scores not found – using placeholder.")
        scores = [{"candidate_id": "spk00", "total_score": 0.5}]

    ids = [s["candidate_id"] for s in scores]
    vals = [s["total_score"] for s in scores]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ids, vals, color="steelblue")
    ax.set_xlabel("Total Score")
    ax.set_title("Candidate Scores")
    ax.set_xlim(0, 1)
    save_figure(fig, save_path)
    logger.info("Figure saved → %s", save_path)


def main() -> None:
    """Entry point for step 03 (figures)."""
    ensure_output_dirs()

    logger.info("Generating report figures …")

    plot_doa_heatmap(FIGURES_DIR / "doa_heatmap.png")
    plot_candidate_scores(FIGURES_DIR / "candidate_scores.png")

    # TODO: Add more figures as the pipeline matures
    #   - Waveform of talker of interest
    #   - VAD timeline per candidate
    #   - Spectrogram before/after enhancement

    logger.info("Step 03 (figures) complete.")


if __name__ == "__main__":
    main()
