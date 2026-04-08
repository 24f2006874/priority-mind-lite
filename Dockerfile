FROM python:3.10-slim

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt pyproject.toml README.md ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY environment.py grader.py inference.py models.py demo.py app.py utils.py ./
COPY openenv.yaml .env.example ./
COPY scripts ./scripts
COPY tests ./tests
COPY server ./server
COPY .huggingface ./.huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct:novita
ENV PORT=7860

# Health check to validate environment is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from environment import PriorityMindEnv; env = PriorityMindEnv('easy'); env.reset(); print(env.state())" || exit 1

# Expose port for Gradio
EXPOSE 7860

# Default command runs the Gradio web interface
CMD ["python", "app.py"]