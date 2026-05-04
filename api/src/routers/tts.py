"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    target_language: str = Query("es", description="Target language code (e.g. 'es')."),
    speaker_wav: str | None = Query(
        None,
        description=(
            "Reference voice WAV path relative to pipeline_data/speakers "
            "(e.g. 'es/default.wav'). When omitted, resolved automatically "
            "from target_language."
        ),
    ),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    *speaker_wav* selects the reference voice for cloning; omitting it auto-resolves
    via ``resolve_speaker_wav(speakers_dir, target_language)``.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    trans_path = trans_dir / f"{title}.json"
    source_path = str(trans_path)

    # Auto-resolve a default reference voice when the caller didn't specify one.
    if speaker_wav is None:
        speaker_wav = resolve_speaker_wav(settings.speakers_dir, target_language)

    # Per-speaker voice assignment: when the translated transcript has diarization
    # speaker labels, map each unique speaker ID to its own reference WAV. Falls
    # back to the language default for any speaker without a dedicated WAV.
    speaker_map: dict[str, str] = {}
    if trans_path.exists():
        translated = json.loads(trans_path.read_text())
        unique_speakers = sorted({
            seg["speaker"] for seg in translated.get("segments", [])
            if seg.get("speaker")
        })
        speaker_map = {
            spk: resolve_speaker_wav(settings.speakers_dir, target_language, spk)
            for spk in unique_speakers
        }

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        speaker_wav=speaker_wav,
        speaker_map=speaker_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
        "speaker_wav": speaker_wav,
        "speaker_map": speaker_map,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
