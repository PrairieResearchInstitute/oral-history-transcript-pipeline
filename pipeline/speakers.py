"""Stage 4: who is speaking when.

Diarization with sherpa-onnx (pyannote segmentation-3.0 + WeSpeaker ResNet34 embeddings,
both ONNX, CPU only), clustered into as many speakers as the YAML lists. Clusters are then
named with simple rules that fit our interviews:

  - the cluster with the most talk time is the interviewee;
  - the remaining clusters are matched to the interviewers in YAML order by who speaks first
    (so list the lead interviewer first).

A person can override the naming with `speaker_map` in the YAML and rerun with
`--redo render`; diarization does not need to run again.

If diarization is turned off (--speakers manual) or fails, every turn is labeled SPEAKER ?.
"""
import os
import subprocess
import time

from .util import PipelineError, log, read_json, warn, write_json

MODELS = os.environ.get("OH_MODELS", "/models")
UNKNOWN = "SPEAKER ?"


def outputs(ctx):
    return [ctx.path("_speakers.json")]


def load_audio(path):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        raise PipelineError(f"could not read {path}: {r.stderr.decode()[:300]}")
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def diarize(wav, num_speakers, threads, models=MODELS):
    import sherpa_onnx as so
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=os.path.join(models, "segmentation.onnx")),
            num_threads=threads),
        embedding=so.SpeakerEmbeddingExtractorConfig(
            model=os.path.join(models, "embedding.onnx"), num_threads=threads),
        clustering=so.FastClusteringConfig(num_clusters=num_speakers),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise PipelineError("sherpa-onnx diarization config is invalid (models missing?)")
    sd = so.OfflineSpeakerDiarization(cfg)
    samples = load_audio(wav)
    last = [0.0]

    def progress(done, total):
        now = time.time()
        if now - last[0] > 30:
            last[0] = now
            log(f"diarization {100.0 * done / max(total, 1):.0f}%")
        return 0

    t0 = time.time()
    result = sd.process(samples, callback=progress).sort_by_start_time()
    segs = [{"start": round(s.start, 3), "end": round(s.end, 3), "cluster": int(s.speaker)} for s in result]
    return segs, time.time() - t0


def name_clusters(segs, speakers, override=None):
    """Return {cluster: name} and a per-cluster summary explaining the choice."""
    stats = {}
    for s in segs:
        st = stats.setdefault(s["cluster"], {"talk_s": 0.0, "first_s": s["start"], "turns": 0})
        st["talk_s"] += s["end"] - s["start"]
        st["first_s"] = min(st["first_s"], s["start"])
        st["turns"] += 1
    names = {}
    interviewees = [p["name"] for p in speakers if p["role"] == "interviewee"]
    interviewers = [p["name"] for p in speakers if p["role"] == "interviewer"]
    by_talk = sorted(stats, key=lambda c: -stats[c]["talk_s"])
    for c, n in zip(by_talk, interviewees):
        names[c] = n
        stats[c]["rule"] = "most talk time -> interviewee"
    rest = sorted((c for c in stats if c not in names), key=lambda c: stats[c]["first_s"])
    for c, n in zip(rest, interviewers):
        names[c] = n
        stats[c]["rule"] = "order of first appearance -> interviewer order in YAML"
    for c in stats:
        if c not in names:
            names[c] = UNKNOWN
            stats[c]["rule"] = "more clusters than speakers"
    for c, n in (override or {}).items():
        if c in stats:
            names[c] = n
            stats[c]["rule"] = "speaker_map in YAML"
    for c in stats:
        stats[c]["name"] = names[c]
        stats[c]["talk_s"] = round(stats[c]["talk_s"], 1)
    return names, stats


def run(ctx):
    out = ctx.path("_speakers.json")
    if ctx.skip("speakers", [out]):
        return
    people = ctx.meta["speakers"]
    if ctx.speaker_method == "manual":
        write_json(out, {"method": "manual", "segments": []})
        log("speaker turns left for a person to fill in (SPEAKER ?)")
        return
    if len(people) == 1:
        write_json(out, {"method": "single", "segments": [
            {"start": 0.0, "end": ctx.probe["duration_s"], "cluster": 0}]})
        return
    log(f"diarizing into {len(people)} speakers (about 1 minute per 10 minutes of audio on 4 cores)")
    try:
        segs, elapsed = diarize(ctx.path("_16k_mono.wav"), len(people), ctx.threads)
    except Exception as e:  # fall back to manual labels rather than lose the transcript
        warn(f"diarization failed, falling back to SPEAKER ?: {e}")
        write_json(out, {"method": "manual", "error": str(e), "segments": []})
        return
    found = len({s["cluster"] for s in segs})
    if found < len(people):
        warn(f"diarization found {found} voices but the YAML lists {len(people)} speakers")
    write_json(out, {
        "method": "sherpa-onnx",
        "models": {"segmentation": "pyannote segmentation-3.0 (ONNX)",
                   "embedding": "WeSpeaker ResNet34-LM VoxCeleb (ONNX)"},
        "num_speakers": len(people),
        "elapsed_s": round(elapsed, 1),
        "segments": segs,
    })
    log(f"diarization took {elapsed / 60:.1f} min, {len(segs)} speech segments")


def load(ctx):
    """Return (diarization segments with names, cluster summary, method)."""
    d = read_json(ctx.path("_speakers.json"))
    if d["method"] == "manual" or not d["segments"]:
        return [], {}, d["method"]
    if d["method"] == "single":
        n = ctx.meta["speakers"][0]["name"]
        return [dict(s, speaker=n) for s in d["segments"]], {0: {"name": n, "rule": "only one speaker"}}, "single"
    names, stats = name_clusters(d["segments"], ctx.meta["speakers"], ctx.meta.get("speaker_map"))
    return [dict(s, speaker=names[s["cluster"]]) for s in d["segments"]], stats, d["method"]
