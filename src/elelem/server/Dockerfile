# Elelem OpenAI API Server
# Docker Hardened Image - Alpine-based, zero CVEs, 147MB
# Requires: docker login dhi.io

# Build stage - Alpine dev image with shell and pip
FROM dhi.io/python:3.12-alpine3.21-dev AS builder

WORKDIR /app

# Copy entire project
COPY . .

# Install dependencies to a virtual env
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        fastapi>=0.104.0 \
        "uvicorn[standard]>=0.24.0" \
        pydantic>=2.5.0 \
        openai>=1.0.0 \
        "sentry-sdk[fastapi]>=1.0.0" \
        httpx \
        pyyaml \
        python-dotenv \
        jsonschema \
        sqlalchemy \
        psycopg2-binary \
        json-repair

# Copy source code to site-packages
# Fix permissions for non-root user in hardened image
RUN cp -r /app/src/elelem /opt/venv/lib/python3.12/site-packages/ && \
    chmod -R a+rX /opt/venv/lib/python3.12/site-packages/elelem

# Runtime stage - Alpine minimal, no shell
FROM dhi.io/python:3.12-alpine3.21

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use venv
ENV PATH="/opt/venv/bin:$PATH"

# Expose API server port only
EXPOSE 8000

# Health check using Python (no curl in hardened image)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

# Run API server only
CMD ["uvicorn", "elelem.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
