# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
# ENV PYTHONUNBUFFERED True

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"


# Copy local code to the container image.
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./

WORKDIR /app
ADD . /app


# Install production dependencies.
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN pip install --no-cache-dir -r requirements.txt


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.

# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

# Expose port
ENV PORT 8080
# Run the application:
CMD ["gunicorn", "app:app", "--config=config.py"]