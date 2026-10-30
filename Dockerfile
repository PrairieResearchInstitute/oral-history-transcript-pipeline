#
# Oral history transcription image: ffmpeg + whisper.cpp + sherpa-onnx diarization + models.
# Build it on the machine that will run it: GGML_NATIVE=ON compiles for this CPU
# (on hualle: AVX2, no AVX-512). The image will crash with "illegal instruction"
# (exit 132) on an older CPU.
#
#   docker build --build-arg MODEL=large-v3-turbo -t oral-history-transcript:large-v3-turbo .
#   docker build --build-arg MODEL=tiny.en        -t oral-history-transcript:tiny.en .

ARG WHISPER_CPP_TAG=v1.9.5

# ---------------------------------------------------------------- whisper.cpp
FROM debian:bookworm-slim AS whisper
ARG WHISPER_CPP_TAG
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${WHISPER_CPP_TAG} https://github.com/ggml-org/whisper.cpp /src \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release \
         -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF \
         -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build /src/build -j"$(nproc)" --target whisper-cli \
    && /src/build/bin/whisper-cli --help > /dev/null

# ---------------------------------------------------------------- models
# Downloaded once at build time and checked against checksums.txt.
# Nothing is downloaded when the pipeline runs.
FROM debian:bookworm-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*
ARG MODEL=large-v3-turbo
ARG MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin
ARG SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download
COPY checksums.txt /models/checksums.txt
WORKDIR /models
RUN set -eu; \
    curl -fSL --retry 3 -o ggml-${MODEL}.bin "${MODEL_URL}"; \
    curl -fSL --retry 3 -O ${SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2; \
    curl -fSL --retry 3 -O ${SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx; \
    grep " ggml-${MODEL}.bin\$" checksums.txt | grep -q . || { echo "no checksum for ggml-${MODEL}.bin in checksums.txt" >&2; exit 1; }; \
    grep -E '^[0-9a-f]{40} ' checksums.txt | grep " ggml-${MODEL}.bin\$" | sha1sum -c -; \
    grep -E '^[0-9a-f]{64} ' checksums.txt | sha256sum -c -; \
    tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2; \
    mv sherpa-onnx-pyannote-segmentation-3-0/model.onnx segmentation.onnx; \
    rm -rf sherpa-onnx-pyannote-segmentation-3-0*; \
    mv wespeaker_en_voxceleb_resnet34_LM.onnx embedding.onnx; \
    echo "${MODEL}" > whisper-model-name

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm
ARG WHISPER_CPP_TAG
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY --from=whisper /src/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=models /models /models
COPY glossary.yml /app/glossary.yml
COPY pipeline /app/pipeline
# Files can arrive owner-only (mode 600); the container runs as the host user, not root.
RUN chmod -R a+rX /app /models
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app WHISPER_CPP_TAG=${WHISPER_CPP_TAG}
WORKDIR /app
ENTRYPOINT ["python", "-m", "pipeline"]
