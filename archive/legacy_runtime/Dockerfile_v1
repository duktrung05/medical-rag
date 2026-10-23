# syntax=docker/dockerfile:1
# Official prebuilt Torch 2.7.1 / CUDA 12.8 runtime (linux/amd64).
# Avoid downloading the multi-GB Torch wheel during pip install.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_RESUME_RETRIES=10 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the image's CUDA-enabled Torch; fail if dependency resolution requires
# a different version instead of silently replacing it.
RUN python -c "import torch; assert torch.__version__.split('+')[0] == '2.7.1'; assert torch.version.cuda == '12.8'; print(torch.__version__, torch.version.cuda)"
COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/constraints.txt /tmp/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements.txt -c /tmp/constraints.txt && python -m pip check

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "src.inspector:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
