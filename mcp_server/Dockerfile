# syntax=docker/dockerfile:1.4
FROM python:3.11-slim AS base

# Install system dependencies with cache mount for apt
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean

# Install uv for package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create app directory
WORKDIR /app

# Set environment variables early
ENV MCP_SERVER_HOST="0.0.0.0"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies with cache mount for uv
RUN --mount=type=cache,target=/root/.cache \
    uv sync --frozen

# Multi-stage build: development stage
FROM base AS development
# Copy application code
COPY graphiti_mcp_server_gemini_ollama.py .
COPY vertex_ai_client.py .
COPY ollama_embedder.py .
COPY ollama_reranker_client.py .

# Production stage
FROM base AS production
# Copy application code
COPY graphiti_mcp_server_gemini_ollama.py .
COPY vertex_ai_client.py .
COPY ollama_embedder.py .
COPY ollama_reranker_client.py .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "graphiti_mcp_server_gemini_ollama.py"]
