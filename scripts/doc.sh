#!/bin/bash

# --- 1. Installation ---
echo "Ensuring Sphinx and dependencies are installed..."
# If your requirements.txt for docs includes -e ., you must be at the root
pip install -r docs/requirements.txt

# --- 2. Build Documentation ---
echo "Building Sphinx documentation..."
# 'sphinx-build' is the command to run the build process.
# -b html: specifies the builder (HTML output)
# docs/source: the source directory
# docs/build/html: the output directory (where the files go)
sphinx-build -b html docs/source docs/build/html

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo " "
    echo "✅ Documentation build successful!"
    echo " "
    # --- 3. Access Instructions ---
    echo "To view the documentation, open the following URL in your web browser:"
    # Use 'open' on macOS or 'xdg-open' on Linux to launch the file
    HTML_PATH="$(pwd)/docs/build/html/index.html"
    echo "file://$HTML_PATH"
    
    # Optional: Automatically open the file (useful for local testing)
    # open "$HTML_PATH"
else
    echo "❌ Documentation build failed."
    exit 1
fi