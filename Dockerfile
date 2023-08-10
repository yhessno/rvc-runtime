FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND="noninteractive"

# Install dependenceis to add PPAs
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA to get Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.9 and pip
RUN apt-get update && \
    apt-get install -y build-essential python3-dev python3.9-distutils python3.9-dev python3.9 curl && \
    apt-get clean && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Set the working directory
WORKDIR /app

# Copy your application files to the container
COPY . .

# Install packages for pyworld
RUN apt-get install -y libsndfile1 libssl3 ffmpeg

# Install python dependencies
RUN pip install -r requirements.txt

# # Expose ports
# EXPOSE 8001

CMD python main.py
