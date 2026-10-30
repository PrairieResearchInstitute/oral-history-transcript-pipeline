"""Stage 1: archival audio copy (no re-encode when possible) and a 16 kHz mono WAV for Whisper."""
import os

from .util import log, run as sh


def archive_name(ctx):
    # AAC (the Canon R6 case) can be copied into .m4a bit for bit. Anything else
    # (PCM, for example) goes to FLAC, which is lossless.
    return ctx.path("_audio.m4a") if ctx.probe["audio_codec"] == "aac" else ctx.path("_audio.flac")


def outputs(ctx):
    return [archive_name(ctx), ctx.path("_16k_mono.wav")]


def run(ctx):
    arch, wav = outputs(ctx)
    if ctx.skip("extract", [arch]):
        pass
    else:
        tmp = arch + ".part" + os.path.splitext(arch)[1]
        codec = ["-c:a", "copy"] if arch.endswith(".m4a") else ["-c:a", "flac"]
        sh(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", ctx.input, "-map", "0:a:0", "-vn", *codec, tmp])
        os.replace(tmp, arch)
    if ctx.skip("extract", [wav]):
        return
    # Read from the small archival copy, not the 20 GB video again.
    tmp = wav + ".part.wav"
    sh(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", arch, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp])
    os.replace(tmp, wav)
    log(f"wrote {os.path.basename(arch)} and {os.path.basename(wav)}")
