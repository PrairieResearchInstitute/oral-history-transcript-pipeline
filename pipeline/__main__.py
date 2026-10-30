"""Entry point inside the container. The host script `transcribe` is the normal way in."""
import argparse
import os
import sys
import time

from . import STAGES, check, extract, render, speakers, transcribe, validate
from .util import PipelineError, log

MODELS = os.environ.get("OH_MODELS", "/models")
# Stages that are cheap always rerun; the expensive ones skip when their outputs exist.
ALWAYS = {"validate", "check", "render"}


class Ctx:
    def __init__(self, a):
        self.input = os.path.abspath(a.input)
        self.meta_path = a.meta
        self.name = os.path.splitext(os.path.basename(self.input))[0]
        self.outdir = a.outdir or os.path.join(os.path.dirname(self.input), self.name)
        self.threads = a.threads
        self.force = a.force
        self.redo = set(a.redo)
        self.speaker_method = a.speakers
        name_file = os.path.join(MODELS, "whisper-model-name")
        self.model_name = open(name_file).read().strip() if os.path.exists(name_file) else "unknown"
        self.model_path = os.path.join(MODELS, f"ggml-{self.model_name}.bin")
        self.meta = self.probe = self.issues = None

    def path(self, suffix, plain=False):
        return os.path.join(self.outdir, suffix if plain else self.name + suffix)

    def skip(self, stage, files):
        if self.force or stage in self.redo:
            return False
        if all(os.path.exists(f) for f in files):
            log(f"{stage}: {', '.join(os.path.basename(f) for f in files)} exists, skipping "
                f"(use --redo {stage} to rerun)")
            return True
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pipeline")
    ap.add_argument("--input", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="rerun every stage (raw Whisper output is kept)")
    ap.add_argument("--redo", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=[],
                    help=f"comma-separated stages to rerun: {','.join(STAGES)}")
    ap.add_argument("--stop-after", choices=STAGES)
    ap.add_argument("--speakers", choices=["auto", "manual"], default="auto")
    a = ap.parse_args(argv)
    bad = [s for s in a.redo if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s) for --redo: {bad}")

    ctx = Ctx(a)
    os.makedirs(ctx.outdir, exist_ok=True)
    mods = {"validate": validate, "extract": extract, "transcribe": transcribe,
            "check": check, "speakers": speakers, "render": render}
    t_all = time.time()
    try:
        for stage in STAGES:
            t0 = time.time()
            log(f"--- {stage}")
            mods[stage].run(ctx)
            dt = time.time() - t0
            if dt > 5:
                log(f"{stage} took {dt / 60:.1f} min")
            if stage == a.stop_after:
                log(f"stopping after {stage} as asked")
                break
    except PipelineError as e:
        log(f"ERROR: {e}")
        return 2
    except KeyboardInterrupt:
        log("interrupted; finished stages are kept and will be skipped next time")
        return 130
    log(f"done in {(time.time() - t_all) / 60:.1f} min; outputs in {ctx.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
