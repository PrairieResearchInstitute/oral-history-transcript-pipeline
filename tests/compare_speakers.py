#!/usr/bin/env python3
"""Compare the pipeline's speaker labels with a hand-labeled transcript (acceptance test 2).

    python3 tests/compare_speakers.py BILL/623A7491/623A7491_speakers.srt BILL/623A7491_transcript.md

The reference is a transcript in this pipeline's format: turns that start with
**[HH:MM:SS] NAME:**. Each subtitle cue is scored against the reference speaker at its
start time. Prints agreement by cue and by word, and a confusion table. Standard library only.
"""
import re
import sys

TURN = re.compile(r"^\*\*\[(\d+):(\d+):(\d+)\] ([^:*]+):\*\*", re.M)
CUE_TIME = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->")


def reference_turns(path):
    text = open(path, encoding="utf-8").read()
    return [(int(h) * 3600 + int(m) * 60 + int(s), n.strip().title()) for h, m, s, n in TURN.findall(text)]


def cues(path):
    blocks = re.split(r"\n\s*\n", open(path, encoding="utf-8").read().strip())
    cur = None
    for b in blocks:
        lines = b.splitlines()
        i = next((k for k, l in enumerate(lines) if "-->" in l), None)
        if i is None:
            continue
        h, m, s, ms = CUE_TIME.search(lines[i]).groups()
        t = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        text = " ".join(lines[i + 1:])
        mm = re.match(r"\[([^\]]+)\]\s*", text)
        if mm:
            cur = mm.group(1).strip().title()
            text = text[mm.end():]
        yield t, cur, len(text.split())


def main(srt, ref):
    turns = reference_turns(ref)
    if not turns:
        sys.exit(f"no **[HH:MM:SS] NAME:** turns found in {ref}")

    def truth(t):
        who = turns[0][1]
        for t0, n in turns:
            if t0 <= t + 0.5:  # reference times are rounded down to the second
                who = n
            else:
                break
        return who

    n = ok = nw = okw = 0
    conf = {}
    for t, got, w in cues(srt):
        tr = truth(t)
        n += 1
        nw += w
        conf[(tr, got)] = conf.get((tr, got), 0) + 1
        if tr == got:
            ok += 1
            okw += w
    print(f"cues agreeing: {ok}/{n} ({ok / n:.1%}); words agreeing: {okw / max(nw, 1):.1%}")
    print(f"{'reference':<22}{'pipeline':<22}cues")
    for (tr, got), c in sorted(conf.items(), key=lambda kv: -kv[1]):
        print(f"{tr:<22}{str(got):<22}{c}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
