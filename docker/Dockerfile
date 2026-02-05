# Start with Python install (slim is smaller and faster)
FROM python:3.11-slim

# Environment safety settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy dependency list and cache for faster rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (bring in Flask, agents, etc.)
COPY . .

# Expose Flask port (runs on port 5000)
EXPOSE 5000

# Run your app
CMD ["python", "app/main.py"]