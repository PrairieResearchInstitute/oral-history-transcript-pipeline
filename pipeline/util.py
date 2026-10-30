import json
import os
import re
import subprocess
import sys
import tempfile


def log(msg):
    print(f"[pipeline] {msg}", file=sys.stderr, flush=True)


def warn(msg):
    print(f"[pipeline] WARNING: {msg}", file=sys.stderr, flush=True)


class PipelineError(Exception):
    pass


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise PipelineError(f"command failed with exit code {r.returncode}: {cmd[0]}")
    return r


def write_text_atomic(path, text):
    """Write via a temp file in the same folder, so an interrupted run never leaves half a file."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path, obj):
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hms(seconds, millis=False):
    seconds = max(0.0, float(seconds))
    if millis:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def duration_hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt_time(t):
    h, m, s, ms = _SRT_TIME.match(t.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000
