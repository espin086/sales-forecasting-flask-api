FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p model

# Copy the trained model and feature list
COPY model/sales_forecast_model.pkl model/
COPY model/feature_list.pkl model/

# Copy the API code
COPY app/ app/

# Set environment variables
ENV FLASK_APP=app/api.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "app/api.py"] 