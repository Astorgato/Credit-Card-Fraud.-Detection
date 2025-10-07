FROM python:3.9-slim

# Metadata
LABEL maintainer="your.email@example.com"
LABEL description="Credit Card Fraud Detection ML Pipeline"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash fraud_user \
    && chown -R fraud_user:fraud_user /app
USER fraud_user

# Expose port (if needed for future web interface)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "src/fraud_detection_models.py"]
