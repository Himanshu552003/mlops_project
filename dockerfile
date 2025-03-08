# Use an official Python 3.11 runtime as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, TensorFlow, and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to avoid interactive prompts and ensure UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Python dependencies with explicit Flask check
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip show flask > /dev/null || (echo "Flask installation failed" && exit 1) \
    && pip list > installed_packages.txt

# Copy the entire project directory
COPY . .

# Ensure the uploads directory exists and is writable
RUN mkdir -p /app/static/uploads && chmod -R 755 /app/static

# Expose the port the app runs on
EXPOSE 5000

# Run the application with error handling
CMD ["sh", "-c", "echo 'Starting application...' && python app.py || echo 'Application failed to start, check logs'"]