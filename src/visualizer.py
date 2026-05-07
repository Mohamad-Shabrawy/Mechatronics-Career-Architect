"""
visualizer.py — Niche Proficiency Radar Chart Generator

This module renders a 6-spoke polar chart using matplotlib's Agg backend,
showing how strongly a user's CV maps to each of the 6 mechatronics niches.
The chart is returned as PNG bytes (never written to disk) so it can be
embedded in the PDF report or served directly over HTTP.

The Agg backend is set BEFORE any pyplot import — this is mandatory for
headless/server environments (no display available). If you move this import,
matplotlib will crash in environments without a screen.

Contract reference: specs/005-skills-gap-analytics/contracts/
Data model:         specs/005-skills-gap-analytics/data-model.md
"""

import io

import matplotlib
matplotlib.use("Agg")  # Must be first — sets non-interactive backend before pyplot loads
import matplotlib.pyplot as plt
import numpy as np

from src.model_io import CANONICAL_NICHES


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def generate_radar_chart(fit_score_result: dict) -> dict:
    """
    Renders a 6-spoke radar chart from a FitScoreResult and returns PNG bytes.

    The chart plots composite_score (0.0–1.0) for each of the 6 canonical niches
    on a polar axis. Niches are arranged in CANONICAL_NICHES order so the layout
    is consistent across calls — the user always sees the same niche in the same
    position.

    The function never raises — all failures are returned as error dicts. This
    keeps the calling pipeline clean: check for "error" key, handle gracefully.

    Parameters
    ----------
    fit_score_result : dict
        A FitScoreResult dict from compute_fit_score(), containing "ranked_niches"
        with 6 entries, each having "niche" and "composite_score".

    Returns
    -------
    dict
        On success — RadarChartResult:
          {
            "chart_image_bytes": bytes,      # PNG-encoded radar chart
            "niche_labels":      list[str],  # 6 niche names in CANONICAL_NICHES order
            "scores":            list[float],# composite scores in same order
            "chart_format":      str,        # always "png"
          }
        On failure — error dict:
          {"error": "<ERROR_CODE>", "message": "<explanation>"}
    """
    try:
        # ── Input validation ──────────────────────────────────────────────────
        # We reject error dicts early to avoid confusing downstream errors.
        if not isinstance(fit_score_result, dict) or "error" in fit_score_result:
            return {
                "error": "INVALID_FIT_RESULT",
                "message": "fit_score_result is an error dict or not a dict",
            }

        ranked = fit_score_result.get("ranked_niches")
        if not isinstance(ranked, list) or len(ranked) != 6:
            return {
                "error": "INVALID_FIT_RESULT",
                "message": (
                    f"ranked_niches must be a list of 6 entries, "
                    f"got {type(ranked).__name__} of length "
                    f"{len(ranked) if isinstance(ranked, list) else 'N/A'}"
                ),
            }

        # Build a lookup map from niche name to composite score.
        # We then reorder by CANONICAL_NICHES so the chart layout is stable.
        score_map = {entry["niche"]: entry["composite_score"] for entry in ranked}
        missing = [n for n in CANONICAL_NICHES if n not in score_map]
        if missing:
            return {
                "error": "INVALID_FIT_RESULT",
                "message": f"ranked_niches is missing entries for: {missing}",
            }

        # ── Build chart data in CANONICAL_NICHES order ────────────────────────
        # Consistent ordering means every chart has the same niche in the same
        # angular position — easier for users to compare charts side by side.
        niche_labels = CANONICAL_NICHES[:]
        scores = [score_map[n] for n in niche_labels]

        # ── Radar chart geometry ──────────────────────────────────────────────
        # A radar chart plots N values on N equally-spaced angular axes.
        # We close the polygon by appending the first point again at the end.
        N = len(niche_labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles_closed = angles + angles[:1]
        scores_closed = scores + scores[:1]

        # ── Render to in-memory PNG ───────────────────────────────────────────
        # Dark-theme palette matching the web UI
        _BG    = '#0B1120'   # panel background
        _LINE  = '#60A5FA'   # sky-blue line
        _FILL  = '#3B82F6'   # fill polygon color
        _GRID  = '#1E293B'   # grid rings / spines
        _LABEL = '#94A3B8'   # spoke tick labels
        _YTICK = '#475569'   # radial value labels

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)

        ax.plot(angles_closed, scores_closed, color=_LINE, linewidth=2.5, zorder=3)
        ax.fill(angles_closed, scores_closed, color=_FILL, alpha=0.18, zorder=2)

        ax.set_xticks(angles)
        ax.set_xticklabels(niche_labels, size=8, color=_LABEL)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.yaxis.set_tick_params(labelcolor=_YTICK, labelsize=6)
        ax.grid(color=_GRID, linewidth=0.8)
        ax.spines['polar'].set_edgecolor(_GRID)

        # Save to a BytesIO buffer — no files written to disk (spec requirement).
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=_BG, edgecolor='none')
        plt.close(fig)  # Release memory — important in long-running server contexts
        buf.seek(0)
        chart_bytes = buf.read()

        return {
            "chart_image_bytes": chart_bytes,
            "niche_labels":      niche_labels,
            "scores":            scores,
            "chart_format":      "png",
        }

    except Exception as e:
        # Safety net — rendering must never crash the calling pipeline.
        return {"error": "RENDER_ERROR", "message": str(e)}
