"""Oral history transcription pipeline: video + interview YAML -> subtitles and a speaker-labeled transcript."""

STAGES = ["validate", "extract", "transcribe", "check", "speakers", "render"]
