# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /opt/app

# Copy the current directory contents into the container at /opt/app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Prometheus Node Exporter
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends prometheus-node-exporter && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose necessary ports
# 7860 for Gradio (default), 8000 for Prometheus metrics, 9100 for Node Exporter
EXPOSE 7860 8000 9100
ENV GRADIO_SERVER_NAME="0.0.0.0"
# Define the command to run both Prometheus Node Exporter and Gradio app
CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python /opt/app/app.py"
