"""The transcript segment format shared by every transcript source."""


def seconds_to_mmss(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def make_segment(start: float, end: float, text: str) -> dict:
    """{start, end, start_str, end_str, text} — the shape claims/ and the UI expect."""
    return {
        "start": start,
        "end": end,
        "start_str": seconds_to_mmss(start),
        "end_str": seconds_to_mmss(end),
        "text": text.strip(),
    }
