"""Fallback transcription: download the audio with yt-dlp and transcribe locally
with faster-whisper (used only when a video has no official captions)."""

import os
import tempfile
import time
from functools import lru_cache

import yt_dlp
from faster_whisper import WhisperModel

from factcheck import config
from factcheck.logging_utils import logger
from factcheck.transcript.segments import make_segment


@lru_cache(maxsize=1)
def get_model():
    """Loaded once per process and reused (the model is slow to initialise)."""
    return WhisperModel(
        config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=os.cpu_count()
    )


def download_audio(video_id: str, out_dir: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_path = os.path.join(out_dir, f"{video_id}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, f"{video_id}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return out_path


def transcribe_with_whisper(video_id: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        t0 = time.time()
        audio_path = download_audio(video_id, tmp_dir)
        logger.info("WHISPER timing | stage=download | duration=%.1fs", time.time() - t0)

        model = get_model()

        t1 = time.time()
        segments_iter, _info = model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = [make_segment(seg.start, seg.end, seg.text) for seg in segments_iter]
        logger.info("WHISPER timing | stage=transcribe | duration=%.1fs", time.time() - t1)
        return segments
