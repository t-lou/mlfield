#!/bin/bash

# Script to stop all Docker containers for the current user

echo "Stopping all Docker containers for user: $(whoami)"
echo ""

# Get all running containers for the current user
containers=$(docker ps -aq)

if [ -z "$containers" ]; then
    echo "No running containers found."
    exit 0
fi

# Count containers
count=$(echo "$containers" | wc -w)
echo "Found $count running container(s). Stopping them..."
echo ""

# Stop all running containers
docker stop $containers
docker rm $containers

if [ $? -eq 0 ]; then
    echo ""
    echo "All containers stopped successfully!"
else
    echo ""
    echo "Error: Failed to stop some containers."
    exit 1
fi
