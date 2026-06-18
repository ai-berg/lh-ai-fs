#!/bin/sh
set -e

# Dependencies are installed at build time (see Dockerfile). The entrypoint only
# starts the server, so boot is fast and pulls nothing from the network.
exec uvicorn main:app --host 0.0.0.0 --port 8002 --reload
