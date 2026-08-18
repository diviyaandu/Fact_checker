import os
import tempfile
import time
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel


@st.cache_resource
def get_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())


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
        print(f"[timing] download: {time.time() - t0:.1f}s")

        model = get_model()

        t1 = time.time()
        segments_iter, _info = model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "start_str": _seconds_to_mmss(seg.start),
                    "end_str": _seconds_to_mmss(seg.end),
                    "text": seg.text.strip(),
                }
            )
        print(f"[timing] transcribe: {time.time() - t1:.1f}s")
        return segments


def _seconds_to_mmss(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"