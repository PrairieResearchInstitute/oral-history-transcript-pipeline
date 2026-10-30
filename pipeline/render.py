"""Stage 5: transcript with speaker turns, review notes, and subtitles with speaker names.

Rules carried over from the hand-made first transcript:
  - one timestamp at the start of each speaker turn;
  - inside a turn, a new paragraph at the first sentence end after about 120 words,
    and a tail shorter than 25 words is merged into the previous paragraph;
  - `corrections` are applied after transcription;
  - the review notes list glossary terms and capitalized names not in the glossary,
    with timestamps, plus turns shorter than 3 seconds.
"""
import hashlib
import os
import re
from collections import OrderedDict

from . import segments as segmod
from . import speakers as spk
from .util import duration_hms, hms, log, read_json, write_json, write_text_atomic

PARA_WORDS = 120
TAIL_WORDS = 25
SHORT_TURN_S = 3.0
NEAREST_S = 3.0          # a Whisper segment with no diarized speech takes the nearest speaker within this
SPLIT_MIN_S = 1.0        # split a segment between speakers only if the second one has this much...
SPLIT_MIN_FRAC = 0.3     # ...and this share of it
AMBIGUOUS_FRAC = 0.6     # below this share for the winning speaker, flag the segment
UNKNOWN = spk.UNKNOWN

# Capitalized words that are not names. Sentence-initial words are handled separately.
NOT_NAMES = set("""
I I'm I've I'll I'd I'd've OK Okay Mr Mrs Ms Dr St Oh Yeah Yes No Well So And But Or The A An
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June
July August September October November December God English American Americans Christmas
""".split())


# ------------------------------------------------------------------ speaker assignment

def _overlaps(t0, t1, dsegs):
    tot = {}
    for d in dsegs:
        if d["end"] <= t0:
            continue
        if d["start"] >= t1:
            break
        o = min(t1, d["end"]) - max(t0, d["start"])
        if o > 0:
            tot[d["speaker"]] = tot.get(d["speaker"], 0.0) + o
    return tot


def _nearest(t0, t1, dsegs):
    best, dist = UNKNOWN, NEAREST_S
    for d in dsegs:
        g = max(d["start"] - t1, t0 - d["end"], 0.0)
        if g < dist:
            best, dist = d["speaker"], g
    return best


def _speaker_at(t0, t1, dsegs):
    ov = _overlaps(t0, t1, dsegs)
    if not ov:
        return _nearest(t0, t1, dsegs), 0.0
    who = max(ov, key=ov.get)
    return who, ov[who] / max(sum(ov.values()), 1e-6)


def assign(segs, dsegs):
    """Split Whisper segments at speaker changes (when token timing allows) and label each piece."""
    pieces = []
    for s in segs:
        if not dsegs:
            pieces.append(dict(s, speaker=UNKNOWN, share=0.0))
            continue
        ov = _overlaps(s["start"], s["end"], dsegs)
        total = sum(ov.values())
        ranked = sorted(ov.items(), key=lambda kv: -kv[1])
        mixed = (len(ranked) > 1 and ranked[1][1] >= SPLIT_MIN_S
                 and ranked[1][1] >= SPLIT_MIN_FRAC * max(total, 1e-6))
        if mixed and len(s["words"]) >= 2:
            pieces += _split_by_words(s, dsegs)
            continue
        who, share = _speaker_at(s["start"], s["end"], dsegs)
        pieces.append(dict(s, speaker=who, share=share))
    return pieces


def _split_by_words(s, dsegs):
    labels = []
    for w in s["words"]:
        who, _ = _speaker_at(w["start"], max(w["end"], w["start"] + 0.05), dsegs)
        labels.append(who)
    # Smooth: a run of one or two words between the same speaker is noise.
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]
    out, cur = [], None
    for w, who in zip(s["words"], labels):
        if cur and cur["speaker"] == who:
            cur["text"] += w["text"]
            cur["end"] = w["end"]
        else:
            cur = {"start": w["start"], "end": w["end"], "text": w["text"], "speaker": who, "share": 0.5, "words": []}
            out.append(cur)
    for p in out:
        p["text"] = p["text"].strip()
    out[0]["start"] = s["start"]
    out[-1]["end"] = s["end"]
    return [p for p in out if p["text"]]


# ------------------------------------------------------------------ text fixes

def apply_corrections(pieces, corrections):
    applied = OrderedDict()
    pats = [(w, r, re.compile(r"(?<![\w])" + re.escape(w) + r"(?![\w])")) for w, r in corrections.items()]
    for p in pieces:
        for wrong, right, pat in pats:
            new, n = pat.subn(right, p["text"])
            if n:
                p["text"] = new
                a = applied.setdefault(wrong, {"right": right, "count": 0, "times": []})
                a["count"] += n
                a["times"].append(p["start"])
    return applied


# ------------------------------------------------------------------ turns and paragraphs

def turns_from(pieces):
    turns = []
    for p in pieces:
        if turns and turns[-1]["speaker"] == p["speaker"]:
            t = turns[-1]
            t["pieces"].append(p)
            t["end"] = p["end"]
        else:
            turns.append({"speaker": p["speaker"], "start": p["start"], "end": p["end"], "pieces": [p]})
    return turns


_SENT_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’)\]])\s+")


def paragraphs(text):
    sents = [s for s in _SENT_END.split(text.strip()) if s]
    paras, cur, n = [], [], 0
    for s in sents:
        cur.append(s)
        n += len(s.split())
        if n >= PARA_WORDS:
            paras.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        tail = " ".join(cur)
        if paras and len(tail.split()) < TAIL_WORDS:
            paras[-1] += " " + tail
        else:
            paras.append(tail)
    return paras


# ------------------------------------------------------------------ review-note scans

def find_terms(pieces, terms):
    found = OrderedDict()
    for term in terms:
        pat = re.compile(r"(?<![\w])" + re.escape(term) + r"(?![\w])", re.IGNORECASE)
        times = [p["start"] for p in pieces for _ in pat.finditer(p["text"])]
        if times:
            found[term] = times
    return found


_CAP_SEQ = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+(?:of|de|van|von|der|la|du)?\s*[A-Z][\w'’.-]*)*")


def find_names(pieces, terms):
    known = {t.lower() for t in terms}
    known_words = {w.lower().strip(".,") for t in terms for w in t.split()}
    mid_sentence_caps = set()
    cands = []
    for p in pieces:
        for sent in re.split(r"(?<=[.!?])\s+", p["text"]):
            for m in _CAP_SEQ.finditer(sent):
                seq = m.group(0).strip(" .'’-")
                words = seq.split()
                initial = m.start() == 0 or sent[:m.start()].strip() in ('"', "“", "'", "(")
                if not initial:
                    mid_sentence_caps.update(words)
                cands.append((seq, words, initial, p["start"]))
    names = OrderedDict()
    for seq, words, initial, t in cands:
        if initial and words and words[0] not in mid_sentence_caps:
            words = words[1:]  # sentence-initial word that is never capitalized elsewhere
        words = [w for w in words if w.strip(".,'’") not in NOT_NAMES]
        if not words:
            continue
        name = " ".join(words)
        if name.lower() in known or all(w.lower().strip(".,'’s") in known_words for w in words):
            continue
        if len(name) < 3 or name.isupper() and len(name) <= 2:
            continue
        names.setdefault(name, []).append(t)
    return names


def _times(ts, n=4):
    s = ", ".join(hms(t) for t in ts[:n])
    return s + (f" and {len(ts) - n} more" if len(ts) > n else "")


# ------------------------------------------------------------------ writers

def _who(meta, role):
    out = []
    for s in meta["speakers"]:
        if s["role"] == role:
            out.append(f"{s['name']} ({s['affiliation']})" if s["affiliation"] else s["name"])
    return ", ".join(out)


def transcript_md(ctx, turns, method):
    m, p = ctx.meta, ctx.probe
    how = {
        "sherpa-onnx": "Speaker turns come from automatic diarization (sherpa-onnx) and have not been checked.",
        "single": "There is one speaker.",
        "manual": "Speaker turns are not labeled yet (SPEAKER ?).",
    }[method]
    lines = [
        f"# {m['title'] or 'Oral History Interview'}", "",
        f"**Interviewee:** {_who(m, 'interviewee')}",
    ]
    if _who(m, "interviewer"):
        lines.append(f"**Interviewers:** {_who(m, 'interviewer')}")
    lines += [
        f"**Date:** {m['date'] or '[to be added]'}",
        f"**Duration:** {duration_hms(p['duration_s'])}",
        f"**Source file:** {p['source_file']}", "",
        f"*Transcribed automatically with Whisper ({ctx.model_name}, "
        f"{('whisper.cpp ' + os.environ.get('WHISPER_CPP_TAG', '')).strip()}). {how} Timestamps mark the start of each speaker turn "
        f"and are accurate to a few seconds. Proper nouns were corrected only where listed in the review notes. "
        f"Raw output is preserved in {ctx.name}.txt / .srt / .vtt / .json.*", "",
        "---", "",
    ]
    for t in turns:
        text = " ".join(pc["text"] for pc in t["pieces"])
        paras = paragraphs(text)
        lines.append(f"**[{hms(t['start'])}] {t['speaker'].upper()}:** {paras[0]}")
        for para in paras[1:]:
            lines += ["", para]
        lines.append("")
    return "\n".join(lines)


def review_md(ctx, turns, pieces, stats, method, applied, terms_found, names_found, issues):
    m = ctx.meta
    L = [f"# Review notes: {m['title'] or ctx.name} ({ctx.name})", "",
         f"Companion to `{ctx.name}_transcript.md`, generated by the pipeline. Timestamps point to the video.", ""]

    L += ["## How speakers were labeled", ""]
    if method == "sherpa-onnx":
        L += ["| Cluster | Labeled as | Talk time | First heard | Rule |", "|---|---|---|---|---|"]
        for c in sorted(stats, key=lambda c: stats[c]["first_s"]):
            s = stats[c]
            L.append(f"| {c} | {s['name']} | {s['talk_s'] / 60:.1f} min | {hms(s['first_s'])} | {s['rule']} |")
        L += ["", "If two people are swapped, set `speaker_map` in the interview YAML "
              "(for example `speaker_map: {0: Emily Benton, 2: Felipe Menanteau}`) and rerun with "
              "`--redo render`. Diarization does not run again.", ""]
    elif method == "manual":
        L += ["Not labeled. Every turn reads SPEAKER ?.", ""]
    else:
        L += ["One speaker.", ""]

    short = [t for t in turns if t["end"] - t["start"] < SHORT_TURN_S and len(turns) > 1]
    amb = [p for p in pieces if 0 < p.get("share", 1) < AMBIGUOUS_FRAC]
    if method == "sherpa-onnx":
        L += ["## Speaker attribution to check", "",
              f"Turns shorter than {SHORT_TURN_S:.0f} seconds, where attribution is least reliable, "
              "and segments where two voices overlap.", ""]
        if short or amb:
            L += ["| Time | Labeled as | Why | Text |", "|---|---|---|---|"]
            rows = [(t["start"], t["speaker"], "short turn", " ".join(p["text"] for p in t["pieces"])) for t in short]
            rows += [(p["start"], p["speaker"], f"overlap ({p['share']:.0%})", p["text"]) for p in amb]
            for t, who, why, text in sorted(rows):
                text = text.replace("|", "/")
                L.append(f"| {hms(t)} | {who} | {why} | {text[:80]}{'...' if len(text) > 80 else ''} |")
        else:
            L.append("None.")
        L.append("")

    L += ["## Corrections applied", ""]
    if applied:
        L += ["| Whisper | Corrected to | Count | Times |", "|---|---|---|---|"]
        for wrong, a in applied.items():
            L.append(f"| {wrong} | {a['right']} | {a['count']} | {_times(a['times'])} |")
    else:
        L.append("None.")
    L.append("")

    L += ["## Glossary terms heard", "", "Check that these are spelled and used correctly.", ""]
    if terms_found:
        L += ["| Term | Count | Times |", "|---|---|---|"]
        for term, ts in terms_found.items():
            L.append(f"| {term} | {len(ts)} | {_times(ts)} |")
    else:
        L.append("None.")
    L.append("")

    L += ["## Names to verify (not in the glossary)", "",
          "Capitalized words and phrases that are not in the glossary. Most proper-noun errors are here. "
          "Add the right spelling to `glossary` and the wrong one to `corrections`, then rerun with `--redo render`.", ""]
    if names_found:
        L += ["| As written | Count | Times |", "|---|---|---|"]
        for name, ts in names_found.items():
            L.append(f"| {name} | {len(ts)} | {_times(ts)} |")
    else:
        L.append("None.")
    L.append("")

    L += ["## Whisper checks", ""]
    if issues:
        for i in issues:
            L.append(f"- **{hms(i['time'])}** {i['kind']}: {i['detail']}")
    else:
        L.append("No repetition loops, no gaps over 15 seconds, and the text runs to the end of the audio.")
    L.append("")
    return "\n".join(L)


def speaker_srt(pieces):
    out, prev = [], None
    for i, p in enumerate(pieces, 1):
        text = p["text"]
        if p["speaker"] != prev:
            text = f"[{p['speaker']}] {text}"
            prev = p["speaker"]
        out += [str(i), f"{hms(p['start'], True)} --> {hms(p['end'], True)}", text, ""]
    return "\n".join(out)


# ------------------------------------------------------------------ hand-edit protection

def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_write(ctx, path, text, state):
    """Do not overwrite a file a person has edited since the pipeline wrote it."""
    key = os.path.basename(path)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            current = _sha(f.read())
        if key in state and current != state[key]:
            alt = path[:-3] + ".new.md" if path.endswith(".md") else path + ".new"
            write_text_atomic(alt, text)
            log(f"{key} was edited by hand; wrote {os.path.basename(alt)} instead")
            return
        if key not in state and current != _sha(text):
            alt = path[:-3] + ".new.md" if path.endswith(".md") else path + ".new"
            write_text_atomic(alt, text)
            log(f"{key} exists and was not written by the pipeline; wrote {os.path.basename(alt)} instead")
            return
    write_text_atomic(path, text)
    state[key] = _sha(text)


def run(ctx):
    segs = segmod.load(ctx.path(""))
    dsegs, stats, method = spk.load(ctx)
    pieces = assign(segs, dsegs)
    applied = apply_corrections(pieces, ctx.meta["corrections"])
    turns = turns_from(pieces)
    terms_found = find_terms(pieces, ctx.meta["terms"])
    names_found = find_names(pieces, ctx.meta["terms"] + list(ctx.meta["corrections"].values()))
    issues = getattr(ctx, "issues", None)
    if issues is None and os.path.exists(ctx.path("_check.json")):
        issues = read_json(ctx.path("_check.json"))["issues"]

    state_path = ctx.path("_render-state.json")
    state = read_json(state_path) if os.path.exists(state_path) else {}
    _safe_write(ctx, ctx.path("_transcript.md"), transcript_md(ctx, turns, method), state)
    _safe_write(ctx, ctx.path("_review_notes.md"),
                review_md(ctx, turns, pieces, stats, method, applied, terms_found, names_found, issues or []), state)
    _safe_write(ctx, ctx.path("_speakers.srt"), speaker_srt(pieces), state)
    write_json(state_path, state)
    n_by = {}
    for t in turns:
        n_by[t["speaker"]] = n_by.get(t["speaker"], 0) + 1
    log(f"{len(turns)} speaker turns: " + ", ".join(f"{k} {v}" for k, v in n_by.items()))
