FROM python:3.11-slim

# Install system dependencies needed by OpenCV (used by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install CPU-only PyTorch FIRST (140MB instead of 800MB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies (ultralytics will skip torch since it's installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

# Reduce memory usage
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
