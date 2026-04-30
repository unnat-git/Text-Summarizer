FROM python:3.9-slim

# Set working directory
WORKDIR /code

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app code
COPY ./app /code/app

# Set environment variables
ENV MODEL_ID="unnat17/Text-Summarizer"

# HF Spaces run on port 7860 by default
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]
