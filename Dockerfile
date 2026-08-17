FROM python:3.11-slim

# System deps for Manim + FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libpango1.0-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create output directories
RUN mkdir -p outputs data saved_videos

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV RENDER=true

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
