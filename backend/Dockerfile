# Multi-stage Dockerfile for FastAPI backend
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy pyproject.toml for better caching
COPY pyproject.toml .
COPY backend/ ./backend/
COPY .git/ ./.git/

# Development stage
FROM base AS development
RUN pip install --no-cache-dir -e .
COPY . .
CMD ["uvicorn", "fivedreg.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 --no-create-home appuser

# Change ownership of the app directory
RUN chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "fivedreg.main:app", "--host", "0.0.0.0", "--port", "8000"]