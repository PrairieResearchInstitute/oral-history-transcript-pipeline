#!/usr/bin/env bash
# End-to-end test with the small tiny.en image. Should finish in under a minute once the image is built.
#   tests/run.sh            # build oral-history-transcript:tiny.en if needed, then run
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUT="$HERE/out"
rm -rf "$OUT" && mkdir -p "$OUT"
cp "$HERE/sample.wav" "$HERE/sample.yml" "$OUT/"

start=$(date +%s)
"$HERE/../transcribe" --input "$OUT/sample.wav" --meta "$OUT/sample.yml" --model "${MODEL:-tiny.en}" --threads 4
first=$(( $(date +%s) - start ))

R="$OUT/sample"
for f in sample_audio.flac sample_16k_mono.wav sample.txt sample.srt sample.vtt sample.json \
         sample_check.json sample_speakers.json sample_transcript.md sample_review_notes.md sample_speakers.srt; do
  [ -s "$R/$f" ] || { echo "FAIL: missing $f"; exit 1; }
done
grep -q "JOHN F. KENNEDY" "$R/sample_transcript.md" || { echo "FAIL: speaker label missing"; exit 1; }
if [ "${MODEL:-tiny.en}" = "tiny.en" ]; then
  grep -qi "ask not what your country" "$R/sample.txt" || { echo "FAIL: unexpected transcript text"; exit 1; }
fi

# A second run must skip the expensive stages.
log=$("$HERE/../transcribe" --input "$OUT/sample.wav" --meta "$OUT/sample.yml" --model "${MODEL:-tiny.en}" --no-build 2>&1)
for s in extract transcribe speakers; do
  echo "$log" | grep -q "$s: .* exists, skipping" || { echo "FAIL: $s did not skip on rerun"; exit 1; }
done
echo "PASS (first run ${first}s)"
