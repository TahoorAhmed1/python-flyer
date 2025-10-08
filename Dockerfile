FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-dotenv  # if you have .env

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Set Flask environment
ENV FLASK_APP=main.py           # Replace 'main.py' with your entry file
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Run Flask
CMD ["flask", "run"]
