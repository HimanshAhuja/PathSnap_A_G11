#!/usr/bin/env bash
# Laptop 1: start the PathSnap API (run from the repo root). Prints this laptop's IP for the UI.
cd "$(dirname "$0")"
[ -d data/evaluator_test ] || { [ -f data/evaluator_test.zip ] && unzip -q -o data/evaluator_test.zip -d data/; }
IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
echo "PathSnap API -> enter  http://$IP:8000  in the UI on Laptop 2"
exec python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
