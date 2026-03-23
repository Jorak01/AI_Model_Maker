# AI Model Suite - Docker Image
# Build: docker build -t ai-model-suite .
# Run:   docker run -p 8000:8000 -p 7860:7860 ai-model-suite
# GPU:   docker run --gpus all -p 8000:8000 ai-model-suite

FROM python:3.11-slim

LABEL maintainer="AI Model Suite"
LABEL description="AI Model Suite with training, chat, image generation, and API"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p checkpoints data plugins logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports: API (8000), Dashboard (5555), Web UI (7860)
EXPOSE 8000 5555 7860

# Default: start API server
CMD ["python", "run.py", "api"]
