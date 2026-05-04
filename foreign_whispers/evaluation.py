"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats
import unicodedata

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    original_translations: list[str] | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard. All scores are in [0, 1] (higher = better).

    Dimensions:
        timing:      Normalized timing accuracy — penalises large duration errors and severe stretch.
        naturalness: Speaking-rate consistency — low variance in chars/s across segments scores higher.
        intelligibility: Fraction of segments that are ACCEPT or MILD_STRETCH (proxy for clean speech).
        semantic_fidelity: Character-level Jaccard similarity between original and (possibly shortened)
                           translations. Only meaningful when original_translations is provided.

    Args:
        metrics: Per-segment timing metrics.
        aligned: Aligned segments from global_align or global_align_dp.
        original_translations: Optional list of pre-shortening translated texts (one per segment),
            used to compute semantic_fidelity. Pass None to skip that dimension.

    Returns:
        Dict with keys: timing, naturalness, intelligibility, semantic_fidelity, overall.
    """
    if not metrics:
        return {k: 0.0 for k in ("timing", "naturalness", "intelligibility", "semantic_fidelity", "overall")}

    # --- Timing (lower error and stretch = higher score) ---
    errors = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    mean_err = _stats.mean(errors)
    pct_severe = sum(1 for a in aligned if a.stretch_factor > 1.4) / len(aligned)
    timing = round(max(0.0, 1.0 - mean_err / 5.0) * (1.0 - pct_severe), 3)

    # --- Naturalness (low speaking-rate variance = higher score) ---
    rates = [
        len(m.translated_text) / m.source_duration_s
        for m in metrics if m.source_duration_s > 0
    ]
    rate_std = _stats.stdev(rates) if len(rates) > 1 else 0.0
    naturalness = round(max(0.0, 1.0 - rate_std / 20.0), 3)

    # --- Intelligibility proxy (fraction of clean segments) ---
    clean = sum(
        1 for m in metrics
        if decide_action(m) in (AlignAction.ACCEPT, AlignAction.MILD_STRETCH)
    )
    intelligibility = round(clean / len(metrics), 3)

    # --- Semantic fidelity (Jaccard similarity of char bigrams) ---
    def _bigrams(text: str) -> set:
        t = unicodedata.normalize("NFC", text.lower())
        return {t[i:i+2] for i in range(len(t) - 1)}

    if original_translations and len(original_translations) == len(metrics):
        sims = []
        for orig, m in zip(original_translations, metrics):
            a, b = _bigrams(orig), _bigrams(m.translated_text)
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
        semantic_fidelity = round(_stats.mean(sims), 3)
    else:
        semantic_fidelity = None

    scores = [timing, naturalness, intelligibility]
    if semantic_fidelity is not None:
        scores.append(semantic_fidelity)
    overall = round(_stats.mean(scores), 3)

    return {
        "timing":            timing,
        "naturalness":       naturalness,
        "intelligibility":   intelligibility,
        "semantic_fidelity": semantic_fidelity,
        "overall":           overall,
    }
