FROM python:3.10-slim

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt pyproject.toml README.md ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct:novita
ENV PORT=7860

# Expose port for Gradio
EXPOSE 7860

# Default command runs the Gradio web interface
CMD ["python", "app.py"]