# oral-history-transcript-pipeline

Turns a video from the PRI oral history project into subtitles and a readable transcript with speaker names. The project records interviews with former survey directors and is run with the University Library.

Everything runs locally in one Docker container started with `--network none`. The recordings contain personal material, so no audio or text leaves the machine. Models are downloaded once, when the image is built.

## Quick start

```bash
git clone https://github.com/PrairieResearchInstitute/oral-history-transcript-pipeline
cd oral-history-transcript-pipeline
cp examples/623A7491.yml ~/Oral-History/Bill-Shilts/623A7491.yml   # then edit it
./transcribe --input ~/Oral-History/Bill-Shilts/623A7491.MP4
```

The first run builds the image (compiles whisper.cpp for this CPU and downloads about 1.7 GB of models). After that, `./transcribe` checks the image is current in a few seconds and starts.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--input FILE` | | Video or audio file |
| `--meta FILE` | `<video name>.yml`, then `interview.yml`, next to the video | Interview metadata |
| `--model NAME` | `large-v3-turbo` | Whisper model; each model is its own image |
| `--threads N` | 6 | CPU threads for Whisper and diarization |
| `--redo STAGES` | | Rerun stages, e.g. `--redo render` or `--redo speakers,render` |
| `--force` | | Rerun every stage |
| `--stop-after STAGE` | | Stop after `validate`, `extract`, `transcribe`, `check` or `speakers` |
| `--speakers manual` | `auto` | Skip diarization; every turn reads `SPEAKER ?` |
| `--no-build` | | Use the existing image without the build check |

## One folder per interview

```
~/Oral-History/
  Bill-Shilts/
    623A7491.MP4
    623A7491.yml                 # interview metadata (you write this)
    623A7491/                    # everything the pipeline writes
      probe.json
      623A7491_audio.m4a         # archival audio, copied from the video without re-encoding
      623A7491_16k_mono.wav      # what Whisper and diarization read
      623A7491.txt .srt .vtt .json   # raw Whisper output, never overwritten
      623A7491_check.json
      623A7491_speakers.json     # who speaks when (diarization clusters)
      623A7491_transcript.md     # the readable transcript
      623A7491_review_notes.md   # what a person should check
      623A7491_speakers.srt      # subtitles with speaker names
```

## Interview metadata

```yaml
title: "Oral History Interview: Bill Shilts"
date: 2026-10-XX
language: en
speakers:                    # interviewee, then interviewers with the lead interviewer first
  - name: Bill Shilts
    role: interviewee
    affiliation: "Former Chief, ISGS; first Executive Director, PRI"
  - name: Emily Benton
    role: interviewer
    affiliation: University Library
  - name: Felipe Menanteau
    role: interviewer
    affiliation: Prairie Research Institute
glossary: [Dick Berg, Mahomet Aquifer, Manteno]     # optional
corrections: {"Mantino": "Manteno"}                  # optional
speaker_map: {0: Emily Benton, 2: Felipe Menanteau}  # optional, only to fix diarization
```

`glossary.yml` in this repo holds PRI-wide names and terms and is merged with each interview's list. Per-interview terms and the speaker names go into the Whisper prompt. Whisper only uses the prompt for its first 30-second window, so most of the value comes from the review notes and `corrections`.

## Stages

| Stage | Tool | Output | Reruns |
|---|---|---|---|
| 0. validate | ffprobe, python | `probe.json`; stops on a bad YAML, missing file, or no audio | always |
| 1. extract | ffmpeg | `_audio.m4a` (or `.flac` if the audio isn't AAC), `_16k_mono.wav` | if missing |
| 2. transcribe | whisper.cpp | `.txt .srt .vtt .json` (the JSON has token timing) | if missing |
| 3. check | python | `_check.json`: repetition loops, gaps over 15 s, text ending early | always |
| 4. speakers | sherpa-onnx | `_speakers.json` | if missing |
| 5. render | python | `_transcript.md`, `_review_notes.md`, `_speakers.srt` | always |

Raw Whisper files are never overwritten. When transcription is rerun, the old files move to `raw-<date>/` first. If someone has edited `_transcript.md` or `_review_notes.md` by hand, render writes `.new.md` next to it instead of replacing it.

## How speakers are assigned

Our recordings have the same signal on both channels, so stereo diarization is not an option. The pipeline uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with the pyannote segmentation-3.0 model and WeSpeaker ResNet34 speaker embeddings, both as ONNX files on CPU. There is no PyTorch in the image and no Hugging Face token is needed.

Diarization groups the audio into as many voices as the YAML lists. The groups are then named by two rules: the voice with the most talk time is the interviewee, and the other voices take the interviewers' names in YAML order by who speaks first. Each Whisper segment takes the speaker it overlaps most. When a segment clearly holds two voices, it is split at the word where the voice changes, using Whisper's token timing.

The review notes show which rule named each voice, and list turns under 3 seconds and segments with overlapping voices, which is where attribution is least reliable. If two people are swapped, set `speaker_map` and run `--redo render`.

On 10-minute excerpts of the Bill Shilts interview, run on four cores of an iMac, diarization took about 70 seconds per 10 minutes of audio, so roughly 11 minutes for the full 93 minutes. Where all three people spoke in the excerpt, 94% of subtitle segments matched the hand-labeled transcript. Short interjections were the misses. The full-length comparison is acceptance test 2 below.

## Requirements

Docker Desktop and the `python3` that comes with macOS (3.8 or newer). `transcribe` uses only the standard library; everything else runs in the container. The image is built for the CPU it is built on (`GGML_NATIVE=ON`), so build it on the machine that runs it. The prebuilt `ghcr.io/ggml-org/whisper.cpp` image uses AVX-512 and crashes on hualle's i7-9750H with exit code 132.

Speed on hualle (6 cores, AVX2): large-v3-turbo runs at about 1.1× real time with 6 threads, so a 93-minute interview takes about 1 hour 45 minutes to transcribe. Giving Docker more CPUs does not help.

## Tests

```bash
python3 -m pytest tests/      # logic only, no Docker; needs pytest and PyYAML on the host
tests/run.sh                  # end to end with a tiny.en image, under a minute after the build
```

`tests/sample.wav` is 11 seconds of John F. Kennedy's 1961 inaugural address (public domain), from whisper.cpp's samples.

Acceptance on hualle:

1. `tests/run.sh` passes.
2. `./transcribe --input ~/Oral-History/Bill-Shilts/623A7491.MP4 --meta examples/623A7491.yml` gives an `.srt` with a segment count within a few percent of the October run (757), and
   `python3 tests/compare_speakers.py ~/Oral-History/Bill-Shilts/623A7491/623A7491_speakers.srt ~/Oral-History/Bill-Shilts/623A7491_transcript.md` shows most turns agreeing with the hand-labeled transcript.
3. Running the same command again skips extract, transcribe and speakers.

## Troubleshooting

- **`apt-get update` fails with "invalid signature" during the build.** Docker Desktop's disk is full. Run `docker builder prune -f` and `docker image prune -f`.
- **Exit code 132 (illegal instruction).** The image was built on a different CPU. Rebuild on this machine: `docker image rm oral-history-transcript:large-v3-turbo`, then run `./transcribe` again.
- **The build is slow to start.** `.dockerignore` keeps everything but the build files out of the build context. Don't remove it, or a 20 GB video in the folder goes to the Docker daemon.

## Versions

whisper.cpp v1.9.5 (`WHISPER_CPP_TAG` build arg), sherpa-onnx 2.0.1, Python 3.12, Debian bookworm. Model checksums are in `checksums.txt`: SHA-1 for the Whisper models (as published in whisper.cpp's `models/README.md`) and SHA-256 for the diarization models.

## Never

- Commit videos, audio, transcripts, models or tokens. `.gitignore` covers the usual file types.
- Send interview audio or text to an outside service.
- Overwrite raw Whisper output.
