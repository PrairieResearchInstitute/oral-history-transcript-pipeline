"""Load Whisper segments from the full JSON (preferred: has token timing) or from the SRT."""
import os
import re

from .util import parse_srt_time, read_json


def _token_words(seg):
    """Group whisper tokens into words with start times (seconds). Skips special tokens like [_BEG_]."""
    words = []
    for tok in seg.get("tokens", []):
        text = tok.get("text", "")
        if not text or (text.startswith("[_") and text.endswith("]")) or text.startswith("<|"):
            continue
        if "offsets" not in tok:  # no token timing: words are useless for splitting
            return []
        t0 = tok.get("offsets", {}).get("from", 0) / 1000.0
        t1 = tok.get("offsets", {}).get("to", 0) / 1000.0
        if text.startswith(" ") or not words:
            words.append({"text": text, "start": t0, "end": t1})
        else:
            words[-1]["text"] += text
            words[-1]["end"] = t1
    return words


def from_json(path):
    d = read_json(path)
    out = []
    for seg in d.get("transcription", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        out.append({
            "start": seg["offsets"]["from"] / 1000.0,
            "end": seg["offsets"]["to"] / 1000.0,
            "text": text,
            "words": _token_words(seg),
        })
    return out


def from_srt(path):
    with open(path, encoding="utf-8") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    out = []
    for b in blocks:
        lines = b.strip().splitlines()
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, z = lines[idx].split("-->")
        text = " ".join(l.strip() for l in lines[idx + 1:]).strip()
        if text:
            out.append({"start": parse_srt_time(a), "end": parse_srt_time(z), "text": text, "words": []})
    return out


def load(prefix):
    if os.path.exists(prefix + ".json"):
        return from_json(prefix + ".json")
    return from_srt(prefix + ".srt")
