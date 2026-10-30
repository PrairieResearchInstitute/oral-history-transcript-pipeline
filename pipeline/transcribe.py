"""Stage 2: whisper.cpp. Writes .txt .srt .vtt and a full .json (segments with token timing).

The raw Whisper files are never overwritten: when this stage is forced, the old ones
are moved to raw-<timestamp>/ first.
"""
import datetime
import os
import shutil

from .util import log, run as sh

EXTS = [".txt", ".srt", ".vtt", ".json"]
PROMPT_MAX_CHARS = 600  # whisper uses at most ~224 prompt tokens; keep well under


def outputs(ctx):
    return [ctx.path(e) for e in EXTS]


def build_prompt(meta):
    """Names and terms in a sentence, which biases Whisper's spelling in the first window."""
    names = [s["name"] for s in meta["speakers"]]
    words = names + [t for t in meta["glossary"] if t not in names]
    prompt = ""
    for w in words:
        nxt = (prompt + ", " + w) if prompt else w
        if len(nxt) > PROMPT_MAX_CHARS:
            break
        prompt = nxt
    title = meta.get("title") or "Oral history interview"
    return f"{title}. {prompt}." if prompt else f"{title}."


def run(ctx):
    outs = outputs(ctx)
    if ctx.skip("transcribe", outs):
        return
    present = [p for p in outs if os.path.exists(p)]
    if present:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        keep = os.path.join(ctx.outdir, f"raw-{stamp}")
        os.makedirs(keep, exist_ok=True)
        for p in present:
            shutil.move(p, os.path.join(keep, os.path.basename(p)))
        log(f"moved previous Whisper output to {os.path.basename(keep)}/")

    prefix = ctx.path("")  # whisper-cli appends the extension
    tmp_prefix = prefix + ".part"
    cmd = ["whisper-cli", "-m", ctx.model_path, "-f", ctx.path("_16k_mono.wav"),
           "-l", ctx.meta["language"], "-t", str(ctx.threads),
           "-otxt", "-osrt", "-ovtt", "-oj", "-ojf", "-of", tmp_prefix,
           "--prompt", build_prompt(ctx.meta)]
    log(f"transcribing with {ctx.model_name}, {ctx.threads} threads "
        f"(about real time for large-v3-turbo on hualle)")
    sh(cmd)
    for e in EXTS:
        os.replace(tmp_prefix + e, prefix + e)
