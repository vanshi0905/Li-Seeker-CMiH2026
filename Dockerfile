# Dockerfile for Li-Seeker MVP
# Critical Minerals Innovation Hackathon 2026 (Problem Statement 01)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies:
# - libgomp1: REQUIRED for XGBoost OpenMP multithreading on Debian-slim
# - curl: REQUIRED for Docker HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install Python dependencies (layer cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Copy project codebase
COPY . /app

# Ensure output and data directories exist
RUN mkdir -p /app/output /app/data

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify dashboard responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default entrypoint for judge evaluation
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
