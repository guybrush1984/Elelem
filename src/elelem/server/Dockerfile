# Elelem OpenAI API Server
# Single-responsibility: API server only (no dashboard)
FROM python:3.11-slim

# Set working directory to project root
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
# Note: This Dockerfile should be run from the project root with:
# docker build -f src/elelem/server/Dockerfile .

# Copy requirements first for better caching
COPY src/elelem/server/requirements-server.txt .

# Copy entire project for editable install
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# Expose API server port only
EXPOSE 8000

# Health check for API server
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server only
CMD ["uvicorn", "elelem.server.main:app", "--host", "0.0.0.0", "--port", "8000"]