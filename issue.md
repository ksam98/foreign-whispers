# Chatterbox TTS CUDA Wedge — Production Stability Bug

## Summary

The Chatterbox TTS container (`travisvn/chatterbox-tts-api:latest`, multilingual model on CUDA) intermittently triggers an unrecoverable CUDA assertion failure mid-run when synthesizing many segments in sequence. Once it fires, every subsequent request returns HTTP 500 until the container is fully restarted. This breaks long-form synthesis runs (Foreign Whispers needs ~98 segments per video).

## Affected stack

- Container: `travisvn/chatterbox-tts-api:latest`
- Underlying lib: `chatterbox-tts` (multilingual variant)
- Model: `chatterbox-tts` LlamaSdpaAttention-based autoregressive token model with voice cloning
- GPU: NVIDIA A100-SXM4-40GB (Lambda Labs)
- Driver: `580.105.08`
- CUDA: `12.4.1` (container) / `13.0` (host)
- PyTorch: bundled in container image
- Calling code: `api/src/services/tts_engine.py` → `ChatterboxClient` (HTTP client to port 8020)

## Reproduction

1. Run the foreign-whispers pipeline end-to-end against a ~7 minute YouTube video that produces ~98 transcript segments (e.g. `GYQ5yGV_-Oc`, *"Strait of Hormuz disruption threatens to shake global economy"*).
2. Hit the TTS endpoint:
   ```
   POST /api/tts/{video_id}?config=c-86ab861&alignment=true
   ```
3. The API engine sends `POST /v1/audio/speech/upload` to chatterbox once per segment with one of `{SPEAKER_00, SPEAKER_01, SPEAKER_02}.wav` as the reference voice.

## Observed failure mode

After **somewhere between 36 and 99 successful segments** (varies stochastically across runs), one request fails. Every request after it also fails with HTTP 500 until the chatterbox container is restarted.

### Failure traces observed

```
✗ TTS generation failed: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the
stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

### Underlying CUDA kernel assertion

The actual assertion (captured from kernel-level log) localises the bug:

```
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex:
  block: [171,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
```

This is `torch.index_select` / embedding lookup with an **out-of-range index** — i.e. a token ID exceeded the embedding table dimension.

## Root cause analysis

The chatterbox autoregressive sampler occasionally produces a token whose ID is `>=` the size of the embedding table it then indexes into. The model is doing something like:

```python
# inside the model's generate loop
next_token = sample(logits, temperature=0.8, cfg_weight=0.5)  # may return out-of-range id
embedding = self.token_embeddings[next_token]                  # CUDA assert fires here
```

This is a **classic sampling instability** in transformer TTS — softmax probabilities can occasionally land on padding/special-token IDs that are valid token slots but are not represented in the embedding matrix, or numerical noise in CFG-guided sampling pushes the argmax into a shifted vocabulary range.

### Why the cascade after first failure

CUDA contexts are per-process. Once a kernel asserts, the entire context is in an error state — every subsequent CUDA call returns the same `device-side assert triggered` regardless of whether its inputs are valid. Recovery requires reinitialising the CUDA context, which in practice means killing and restarting the Python process (i.e. `docker compose restart chatterbox-gpu`).

### Why it's stochastic

The sampler is non-deterministic (`temperature=0.8`, no fixed seed exposed). Different runs with identical inputs will produce different token sequences, and only some of those sequences contain the rogue token. This is why we observed the wedge at request 48 in one run, request 36 in another, and request 63 in a third.

## Workarounds attempted

| Attempt | Outcome |
|---------|---------|
| Restart container, retry | Sometimes works (baseline TTS completed all 99 on second attempt). Sometimes fails again at a different request number. |
| Set `FW_TTS_WORKERS=1` (serial synthesis from API engine) | Did not propagate into container env; multiple parallel HTTP calls still hit chatterbox. Inconclusive whether this would help — chatterbox processes them serially internally. |
| `nvidia-smi` shows healthy GPU between runs | Confirms the wedge is process-state, not hardware. |
| Wait between requests | Not attempted. |
| Pin transformers / chatterbox version | Not attempted — would require rebuilding the upstream container. |

## What would actually fix it

These are real fixes (none cheap):

1. **Sampler clamping** — wrap `chatterbox`'s token sampling so any token ID `>= vocab_size` is rejected and resampled. Requires forking `chatterbox-tts`.
2. **Process-level recovery** — wrap the chatterbox HTTP server so that on any 500 it auto-restarts the model worker before responding. Requires forking the `chatterbox-tts-api` server.
3. **Set a fixed seed** — possibly avoids the bad token sequence on this specific input, but is not a real fix.
4. **`CUDA_LAUNCH_BLOCKING=1`** — would yield a precise stack trace identifying the exact line in chatterbox source, enabling a targeted patch. Has a perf cost but worth running once for diagnosis.
5. **Pin `transformers` to a known-good version** — chatterbox emits a `FutureWarning` about LlamaSdpaAttention changes in transformers v5; possible the bug is masked or surfaced by specific transformers internals.

For a class project the cost-benefit of any of these is unfavourable; the workaround is "restart and retry."

## Impact on this project

- **Notebook 5 metrics are unaffected** — the regression model was fit on the per-segment `raw_duration_s` values that *do* get written to the align.json sidecar before the wedge, so all four alignment tasks completed with valid data.
- **Baseline TTS run completed cleanly** on second attempt, producing a working dubbed `.mp4` with multiple per-speaker cloned voices. This is the deliverable sample I/O.
- **Aligned TTS run repeatedly hits the wedge** at varying request numbers (36, 63, etc.), so we have not been able to produce a fully-time-stretched aligned `.mp4`. The aligned mode is demonstrated quantitatively in Notebook 5's scorecard (timing score 0.77 → 0.80, intelligibility 0.80 → 0.90) rather than as a final stitched video.

## Status

**Workaround in production:** the `tts_engine.py` retry path treats individual segment failures as silence (returns `None` from `_synthesize_raw`, triggers `AudioSegment.silent(target_ms)` fallback in `_postprocess_segment`). This degrades a wedged run to a partially-silent WAV rather than crashing the entire pipeline, but it's not a solution — once the wedge fires every subsequent segment becomes silence.

**Proper fix:** out of scope for this project. Filed as documentation + workaround.
