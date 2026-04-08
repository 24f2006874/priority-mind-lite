FROM python:3.10-slim

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt pyproject.toml README.md ./

# Install git (needed for any git-based dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
ENV PORT=7860

# Health check to validate environment is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()" || exit 1

# Expose port for Gradio + OpenEnv endpoints
EXPOSE 7860

# Default command runs the combined Gradio + OpenEnv app
CMD ["python", "app.py"]
