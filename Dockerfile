FROM mcr.microsoft.com/devcontainers/python:3.10

ARG _DEVCONTAINER_FEATURES
RUN ${_DEVCONTAINER_FEATURES}

RUN apt-get update && apt-get install -y git-lfs libgl1

# Git LFS initialisieren
RUN git lfs install

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


