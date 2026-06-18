FROM node:20-slim

WORKDIR /app

# Install dependencies at build time (as root, into /app/node_modules) instead
# of at boot. This mirrors the backend: deps live in an immutable layer, the
# entrypoint only starts the dev server, and the anonymous node_modules volume
# is seeded from the image — so the hardened non-root runtime (user 1000) can
# read them without hitting EACCES trying to write into a root-owned volume.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Vite writes its dep-optimizer cache into node_modules/.vite at runtime, so the
# non-root runtime user (1000, set in docker-compose.override.yml) must own the
# tree — including the node_modules that seeds the anonymous volume.
RUN chown -R 1000:1000 /app

ENTRYPOINT ["./entrypoint.sh"]
