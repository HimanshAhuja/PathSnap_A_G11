#!/usr/bin/env bash
# Laptop 2: serve the evaluator UI, then open http://localhost:8080 in a browser.
cd "$(dirname "$0")/ui" && echo "Open http://localhost:8080" && exec python3 -m http.server 8080
