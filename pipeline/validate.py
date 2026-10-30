"""Stage 0: fail early on a bad YAML, a missing file, or no audio stream. Writes probe.json."""
import json
import os
import subprocess

from . import meta as metamod
from .util import PipelineError, log, write_json


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise PipelineError(f"ffprobe could not read {path}: {r.stderr.strip()}")
    return json.loads(r.stdout)


def run(ctx):
    if not os.path.isfile(ctx.input):
        raise PipelineError(f"input not found: {ctx.input}")
    ctx.meta = metamod.load(ctx.meta_path)
    p = probe(ctx.input)
    audio = [s for s in p.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise PipelineError(f"{ctx.input} has no audio stream")
    a = audio[0]
    duration = float(a.get("duration") or p.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise PipelineError(f"could not read the duration of {ctx.input}")
    ctx.probe = {
        "source_file": os.path.basename(ctx.input),
        "duration_s": duration,
        "audio_codec": a.get("codec_name"),
        "audio_channels": a.get("channels"),
        "audio_sample_rate": int(a.get("sample_rate") or 0),
        "format": p.get("format", {}).get("format_name"),
        "size_bytes": int(p.get("format", {}).get("size") or 0),
    }
    write_json(ctx.path("probe.json", plain=True), ctx.probe)
    sp = ", ".join(f"{s['name']} ({s['role']})" for s in ctx.meta["speakers"])
    log(f"{ctx.probe['source_file']}: {duration / 60:.1f} min, {a.get('codec_name')} "
        f"{a.get('channels')}ch {a.get('sample_rate')} Hz; speakers: {sp}")
