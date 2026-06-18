#!/bin/sh
set -e

# Dependencies are installed at build time (see Dockerfile). The entrypoint only
# starts the Vite dev server, so boot is fast and writes nothing to node_modules.
exec npm run dev -- --host 0.0.0.0
