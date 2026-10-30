"""Stage 3: sanity checks on the raw Whisper output. Warnings only; never fails the run."""
from . import segments as segmod
from .util import hms, warn, write_json

GAP_S = 15.0
END_TOLERANCE_S = 30.0
REPEAT_RUN = 3


def find_issues(segs, duration):
    issues = []
    if not segs:
        return [{"kind": "empty", "time": 0.0, "detail": "Whisper produced no text"}]
    # Repetition loops: the same text in REPEAT_RUN or more consecutive segments.
    run_start = 0
    for i in range(1, len(segs) + 1):
        same = i < len(segs) and segs[i]["text"].strip().lower() == segs[run_start]["text"].strip().lower()
        if not same:
            n = i - run_start
            if n >= REPEAT_RUN:
                issues.append({"kind": "repeat", "time": segs[run_start]["start"],
                               "detail": f"\"{segs[run_start]['text'][:60]}\" repeated {n} times"})
            run_start = i
    prev_end = 0.0
    for s in segs:
        if s["start"] - prev_end > GAP_S:
            issues.append({"kind": "gap", "time": prev_end,
                           "detail": f"no text for {s['start'] - prev_end:.0f} s (until {hms(s['start'])})"})
        prev_end = max(prev_end, s["end"])
    if duration and duration - prev_end > END_TOLERANCE_S:
        issues.append({"kind": "end", "time": prev_end,
                       "detail": f"last text ends at {hms(prev_end)} but the audio runs to {hms(duration)}"})
    return issues


def run(ctx):
    segs = segmod.load(ctx.path(""))
    issues = find_issues(segs, ctx.probe["duration_s"])
    for i in issues:
        warn(f"{hms(i['time'])} {i['kind']}: {i['detail']}")
    write_json(ctx.path("_check.json"), {"segments": len(segs), "issues": issues})
    ctx.issues = issues
