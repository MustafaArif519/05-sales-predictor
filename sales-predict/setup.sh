#!/bin/bash

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv could not be found"
    echo "Installing virtualenv..."
    pip install virtualenv
fi

# Create a virtual environment
echo "Creating Virtual Environment..."
virtualenv venv &> /dev/null
echo "Created Virtual Environment"

# Activate the virtual environment
source venv/bin/activate &> /dev/null


# Check if requirements.txt exists in the current directory
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt &> /dev/null
    echo "Installed dependencies from requirements.txt"
else
    echo "requirements.txt not found in the current directory."
fi

# Deactivate the virtual environment
deactivate

# Activate the running script of sales-predict program
# chmod +x scripts/running.sh