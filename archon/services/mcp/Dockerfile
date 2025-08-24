FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python path to include the current directory
ENV PYTHONPATH="/app"

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional required packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    python-dotenv \
    httpx \
    sse-starlette

# Copy the MCP server files
COPY . .

# Create a simple __init__.py if it doesn't exist
RUN touch /app/__init__.py

# Expose port for MCP server
EXPOSE 8100

# Command to run the MCP server
CMD ["python", "-m", "uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8100", "--reload", "--reload-dir", "/app"]