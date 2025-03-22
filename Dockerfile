FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY make.py .
COPY config.json.template .

# Create a volume for input/output books
VOLUME /books

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["python", "make.py"]

# Default command (will be appended to ENTRYPOINT)
CMD ["--help"]