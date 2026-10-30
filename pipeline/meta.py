"""Interview metadata: load and validate the per-interview YAML and merge it with the shared glossary."""
import datetime
import os

import yaml

from .util import PipelineError

ROLES = {"interviewee", "interviewer"}
SHARED_GLOSSARY = os.environ.get("OH_GLOSSARY", os.path.join(os.path.dirname(__file__), "..", "glossary.yml"))


def _as_list(v, what):
    if v is None:
        return []
    if not isinstance(v, list):
        raise PipelineError(f"'{what}' must be a list")
    return [str(x).strip() for x in v if str(x).strip()]


def _as_map(v, what):
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise PipelineError(f"'{what}' must be a mapping of wrong -> right")
    return {str(k): str(x) for k, x in v.items()}


def load_shared_glossary(path=SHARED_GLOSSARY):
    if not os.path.exists(path):
        return [], {}
    with open(path, encoding="utf-8") as f:
        g = yaml.safe_load(f) or {}
    terms = []
    for key, val in g.items():
        if key == "corrections":
            continue
        terms += _as_list(val, f"glossary.yml:{key}")
    return terms, _as_map(g.get("corrections"), "glossary.yml:corrections")


def load(path):
    """Return a normalized dict. Raises PipelineError with a readable message on any problem."""
    if not os.path.exists(path):
        raise PipelineError(f"metadata file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise PipelineError(f"{path} is not valid YAML: {e}")
    if not isinstance(raw, dict):
        raise PipelineError(f"{path} must be a YAML mapping")

    problems = []
    title = str(raw.get("title") or "").strip() or None
    date = raw.get("date")
    if isinstance(date, (datetime.date, datetime.datetime)):
        date = date.isoformat()
    date = str(date).strip() if date else None
    language = str(raw.get("language") or "").strip()
    if not language:
        problems.append("'language' is required (e.g. en)")

    speakers = []
    names = set()
    for i, s in enumerate(raw.get("speakers") or []):
        if not isinstance(s, dict) or not s.get("name"):
            problems.append(f"speakers[{i}] needs a name")
            continue
        name = str(s["name"]).strip()
        role = str(s.get("role") or "").strip().lower()
        if role not in ROLES:
            problems.append(f"speaker '{name}': role must be one of {sorted(ROLES)}")
        if name.lower() in names:
            problems.append(f"speaker '{name}' is listed twice")
        names.add(name.lower())
        speakers.append({"name": name, "role": role, "affiliation": str(s.get("affiliation") or "").strip() or None})
    if not speakers:
        problems.append("'speakers' must list at least one speaker")
    elif not any(s["role"] == "interviewee" for s in speakers):
        problems.append("one speaker must have role: interviewee")

    # Optional manual override after a person has checked the diarization:
    #   speaker_map: {0: Emily Benton, 1: Bill Shilts, 2: Felipe Menanteau}
    speaker_map = {}
    for k, v in (raw.get("speaker_map") or {}).items():
        try:
            k = int(k)
        except (TypeError, ValueError):
            problems.append(f"speaker_map key '{k}' must be a cluster number")
            continue
        if str(v).lower() not in names and str(v) != "SPEAKER ?":
            problems.append(f"speaker_map {k}: '{v}' is not one of the speakers")
        speaker_map[k] = str(v)

    try:
        glossary = _as_list(raw.get("glossary"), "glossary")
        corrections = _as_map(raw.get("corrections"), "corrections")
    except PipelineError as e:
        problems.append(str(e))
        glossary, corrections = [], {}

    if problems:
        raise PipelineError(f"problems in {path}:\n  - " + "\n  - ".join(problems))

    shared_terms, shared_corr = load_shared_glossary()
    merged_corr = dict(shared_corr)
    merged_corr.update(corrections)  # per-interview wins
    seen, terms = set(), []
    for t in [s["name"] for s in speakers] + glossary + shared_terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)

    return {
        "title": title,
        "date": date,
        "language": language,
        "speakers": speakers,
        "speaker_map": speaker_map,
        "glossary": glossary,           # per-interview terms (used in the whisper prompt first)
        "terms": terms,                 # speakers + per-interview + shared, for review notes
        "corrections": merged_corr,
    }
