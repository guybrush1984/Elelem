# Elelem OpenAI API Server
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

# Expose ports - API server on 8000, Dashboard on 8501
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run both services using shell script
CMD ["sh", "-c", "uvicorn elelem.server.main:app --host 0.0.0.0 --port 8000 & streamlit run src/elelem/server/dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true & wait"]