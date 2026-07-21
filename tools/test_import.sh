#!/bin/bash

##########################################################################
## Run the Python scripts as main to check the imports and basic syntax. #
##########################################################################

set -euo pipefail

# Load the envrc
. ./.envrc

# Glob all Python scripts in components
TARGET="components"

files=$(find $TARGET -name "*.py")
for file in ${files}; do
    echo "Checking ${file}"
    runpy ${file}
done