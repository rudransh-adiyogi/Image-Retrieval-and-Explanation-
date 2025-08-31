FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, models, index, metadata, static, templates
COPY . .

# Expose API port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
