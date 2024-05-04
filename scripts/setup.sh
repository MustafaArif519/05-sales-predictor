#!/bin/bash

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv could not be found"
    echo "Installing virtualenv..."
    pip install virtualenv
fi

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Check if requirements.txt exists in the current directory
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found in the current directory."
fi

# Deactivate the virtual environment
deactivate