"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds.

    Linear regression fit on 60 ground-truth Chatterbox TTS segments
    (Spanish, Strait of Hormuz video). MAE 0.44s vs 1.24s for the old heuristic.
    """
    # Old syllable-rate heuristic (MAE 1.24s):
    # return _count_syllables(text) / 4.5
    syllables = _count_syllables(text)
    words = len(text.split())
    chars = len(text)
    return max(0.1, 0.0648 * syllables + 0.1144 * words + 0.0266 * chars + 0.5891)


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    beam_width:      int   = 5,
) -> list[AlignedSegment]:
    """Beam-search global timeline alignment.

    Improves on the greedy ``global_align()`` by maintaining *beam_width*
    candidate schedules simultaneously and selecting the one with the lowest
    total penalty at the end.  This allows silence gaps to be allocated to
    the segments that need them most rather than the first segment that asks.

    The penalty for each candidate schedule is::

        Σ  overflow_s²  +  |cumulative_drift|  +  10 * (stretch > max_stretch)

    where the 10x term heavily penalises severe time-stretching.

    Requires ``silence_regions`` to be non-empty to show improvement over
    the greedy baseline — without gaps there is nothing to reallocate.
    Use inter-segment transcript gaps as a lightweight substitute for VAD output.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output or synthetic inter-segment gaps.
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.
        beam_width: Number of candidate schedules to maintain.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _best_gap(end_s: float, used: dict) -> tuple[float, int]:
        for i, r in enumerate(silence_regions):
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                remaining = (r["end_s"] - r["start_s"]) - used.get(i, 0.0)
                if remaining > 0.05:
                    return remaining, i
        return 0.0, -1

    def _seg_penalty(action: "AlignAction", overflow: float, drift_added: float, stretch: float) -> float:
        # drift_added = drift this segment introduces (not cumulative). Otherwise every
        # gap_shift would be billed N times via downstream segments and beam search
        # would never take a gap_shift. Unfixed overflow (REQUEST_SHORTER / FAIL) is
        # the real cost to penalise heavily.
        pen = 0.3 * abs(drift_added)
        if stretch > max_stretch:
            pen += 10.0
        if action == AlignAction.REQUEST_SHORTER:
            pen += 5.0 * overflow + 2.0
        elif action == AlignAction.FAIL:
            pen += 20.0 * overflow + 10.0
        return pen

    # Each beam entry: (total_penalty, cumulative_drift, segments, silence_used)
    beam: list[tuple[float, float, list[AlignedSegment], dict]] = [
        (0.0, 0.0, [], {})
    ]

    for m in metrics:
        candidates = []

        for tot_pen, drift, segs, sil_used in beam:
            avail_gap, gap_idx = _best_gap(m.source_end, sil_used)

            # Option A: no gap — greedy-style decision
            action_a  = decide_action(m, available_gap_s=0.0)
            stretch_a = min(m.predicted_stretch, max_stretch) if action_a == AlignAction.MILD_STRETCH else 1.0
            ss_a      = m.source_start + drift
            se_a      = ss_a + m.source_duration_s
            pen_a     = tot_pen + _seg_penalty(action_a, m.overflow_s, 0.0, stretch_a)
            candidates.append((pen_a, drift, segs + [AlignedSegment(
                index=m.index, original_start=m.source_start, original_end=m.source_end,
                scheduled_start=ss_a, scheduled_end=se_a, text=m.translated_text,
                action=action_a, gap_shift_s=0.0, stretch_factor=stretch_a,
            )], sil_used))

            # Option B: borrow from the next silence gap (if useful)
            if avail_gap > 0 and m.overflow_s > 0:
                borrow    = min(m.overflow_s, avail_gap)
                action_b  = decide_action(m, available_gap_s=borrow)
                if action_b == AlignAction.GAP_SHIFT:
                    new_drift = drift + borrow
                    ss_b      = m.source_start + drift
                    se_b      = ss_b + m.source_duration_s + borrow
                    pen_b     = tot_pen + _seg_penalty(AlignAction.GAP_SHIFT, 0.0, borrow, 1.0)
                    new_sil   = {**sil_used, gap_idx: sil_used.get(gap_idx, 0.0) + borrow}
                    candidates.append((pen_b, new_drift, segs + [AlignedSegment(
                        index=m.index, original_start=m.source_start, original_end=m.source_end,
                        scheduled_start=ss_b, scheduled_end=se_b, text=m.translated_text,
                        action=AlignAction.GAP_SHIFT, gap_shift_s=borrow, stretch_factor=1.0,
                    )], new_sil))

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    _, _, best_segs, _ = beam[0]
    return best_segs
