# Base image — slim Python 3.9
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# So rebuilds are fast if only code changes
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]