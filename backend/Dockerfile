FROM python:3.11-slim

WORKDIR /app

# Install dependencies at build time (as root, into the global site-packages on
# PATH) instead of at boot. This keeps deps in an immutable, scannable layer,
# removes the per-boot download surface, and makes the image work under the
# hardened non-root runtime (see the Security note in README.md). Hot-reload of
# source still works because the source tree is bind-mounted at runtime.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

ENTRYPOINT ["./entrypoint.sh"]
