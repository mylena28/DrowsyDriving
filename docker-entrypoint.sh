#!/bin/bash
set -e

# Runs as root — fix ownership on mounted volumes before dropping privileges
mkdir -p /app/logs /tmp/libcamera /tmp/Ultralytics
chown -R salte:salte /app/logs /tmp/libcamera /tmp/Ultralytics 2>/dev/null || true

# Drop to salte user and run the main command
exec runuser -u salte -- "$@"
