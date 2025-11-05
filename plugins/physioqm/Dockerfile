# PhysioQM Plugin - Dockerfile
# ================================
# Multi-stage build for quantum-informed PBPK prediction
#
# Author: Agourakis
# Created: 2025-10-31T01:30:00Z
# Purpose: Real science, not just paper
# Darwin Indexed: Auto

FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================================
# Stage 1: Build dependencies
# ============================================================================

FROM base as builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================

FROM base as runtime

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ /app/

# Copy proto-generated files from Darwin Core
COPY --from=darwin-core:2.0.0 /app/protos/physioqm /app/protos/physioqm

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GRPC_PORT=50052
ENV QDRANT_HOST=darwin-qdrant
ENV QDRANT_PORT=6333

# Expose gRPC port
EXPOSE 50052

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50052'); print('OK')"

# Run gRPC server
CMD ["python", "-m", "app.main"]

