#!/bin/bash

# Create books directory if it doesn't exist
mkdir -p books

# Create config from template if it doesn't exist
if [ ! -f config.json ]; then
  cp config.json.template config.json
  echo "Created config.json from template. Please edit it to add your API key."
fi

# Try with docker compose (newer version) first, fall back to docker-compose
if command -v docker &> /dev/null; then
  if docker compose version &> /dev/null; then
    echo "Using docker compose"
    docker compose run --rm epub-translator "$@"
  elif command -v docker-compose &> /dev/null; then
    echo "Using docker-compose"
    docker-compose run --rm epub-translator "$@"
  else
    echo "Error: Neither 'docker compose' nor 'docker-compose' commands are available."
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
  fi
else
  echo "Error: Docker is not installed or not in your PATH."
  echo "Please install Docker: https://docs.docker.com/get-docker/"
  exit 1
fi