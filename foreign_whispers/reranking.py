"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import os

from ollama import Client
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_CHARS_PER_SECOND = 15
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "aya:8b")
_ollama = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

class _Candidate(BaseModel):
    text: str
    brevity_rationale: str

class _CandidateList(BaseModel):
    candidates: list[_Candidate]


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.



    Returns:
        List of ``TranslationCandidate`` items sorted shortest first.
    """

    # STUDENT IMPLEMENTATION NOTE
    #
    # Implementation uses Ollama to run a local LLM (aya:8b) that generates shorter
    # translation candidates fitting within the character budget.  Ollama runs as a local
    # server on port 11434 — the Python package is an HTTP client that talks to it, so
    # no cloud dependency or API key needed.
    #
    # aya:8b (Cohere) was chosen because it's purpose-built for multilingual tasks across
    # 23+ languages, making it a better fit for translation than a general-purpose model
    # of the same size.  At ~5GB it also fits comfortably on a MacBook Pro.
    #
    # I wrote the initial version of the code without the use of AI by following the docs here:
    # https://ollama.com/docs/python-client. I used AI for prompt iteration and schema enforcement for
    # structured output: the Ollama docs (https://github.com/ollama/ollama-python?tab=readme-ov-file#structured-outputs)
    # show that passing format='json' only nudges the model toward JSON without enforcing
    # any specific shape.
    #
    # TODO: if needed add another option of calling Claude via API for translation re-ranking

    char_limit = int(target_duration_s * _CHARS_PER_SECOND)
    targets = {
        "minimal": char_limit,
        "moderate": int(char_limit * 0.90),
        "aggressive": int(char_limit * 0.75),
    }

    prompt = (
        f"You are a professional subtitle translator. Produce shorter translations "
        f"of the segment below that fit within a strict character budget.\n\n"
        f"Source text: \"{source_text}\"\n"
        f"Baseline translation: \"{baseline_es}\" ({len(baseline_es)} chars — too long)\n"
        f"Character budget: {char_limit} characters ({target_duration_s:.1f}s × 15 chars/s)\n"
    )
    if context_prev:
        prompt += f"Previous segment: \"{context_prev}\"\n"
    if context_next:
        prompt += f"Next segment: \"{context_next}\"\n"
    prompt += (
        f"\nProduce exactly 3 candidates at these target lengths:\n"
        f"- Candidate 1: ~{targets['minimal']} characters (minimal compression, just under budget)\n"
        f"- Candidate 2: ~{targets['moderate']} characters (moderate compression)\n"
        f"- Candidate 3: ~{targets['aggressive']} characters (aggressive compression)\n\n"
        "For each candidate, include a brevity_rationale field explaining what was "
        "shortened or omitted to reach the target length (the brevity rationale must be in english)."
    )

    response = _ollama.chat(
        model=_OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=_CandidateList.model_json_schema(),
    )

    parsed = _CandidateList.model_validate_json(response.message.content)
    candidates = [
        TranslationCandidate(
            text=c.text,
            char_count=len(c.text),
            brevity_rationale=c.brevity_rationale,
        )
        for c in parsed.candidates
    ]
    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: %.1fs budget (%d char limit), %d candidates returned.",
        target_duration_s,
        char_limit,
        len(candidates),
    )
    return candidates
