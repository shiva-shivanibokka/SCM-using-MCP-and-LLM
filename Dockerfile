# Dockerfile — HuggingFace Spaces (Docker SDK, CPU)
# HF Spaces requires the Dockerfile at the repo root. Build context is the
# repository root, so all paths below are root-relative.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    BYPASS_MCP_HTTP=true

WORKDIR /app

# Install torch CPU wheels first (smaller, no CUDA)
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0"

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY . /app

# Generate the dataset at build time so the API has data on first boot
RUN python data/generate_data.py

# Pre-download Chronos so the first request is fast (not at runtime)
RUN python -c "import torch; from chronos import ChronosPipeline; \
ChronosPipeline.from_pretrained('amazon/chronos-t5-small', device_map='cpu', torch_dtype=torch.float32)"

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
