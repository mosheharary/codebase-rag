# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY app.py .
COPY Neo4jGraphClient.py .
COPY OpensearchClient.py .
COPY LlmClient.py .
COPY .env .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Define environment variable (optional)
ENV PYTHONUNBUFFERED=1

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
