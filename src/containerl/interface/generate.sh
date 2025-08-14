#!/bin/bash

# Ensure we're in the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if grpcio-tools is installed
if ! python -c "import grpc_tools.protoc" &> /dev/null; then
    echo "grpcio-tools not found. Installing..."
    if ! pip install grpcio-tools; then
        echo "Failed to install grpcio-tools"
        exit 1
    fi
fi

# Generate the Python code from proto file
if ! python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    ./proto.proto; then
    echo "Failed to generate gRPC Python code"
    exit 1
fi

echo "gRPC Python code generated successfully."

# Create an __init__.py file if it doesn't exist to make the directory a proper package
touch __init__.py

echo "Done!"
