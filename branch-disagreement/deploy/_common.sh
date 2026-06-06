#!/usr/bin/env bash
# Shared helpers for the deploy scripts. Sourced, not executed directly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
CONFIG_FILE="$HERE/config.env"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: $CONFIG_FILE not found." >&2
  echo "Copy deploy/config.env.example to deploy/config.env and fill it in." >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a; . "$CONFIG_FILE"; set +a

: "${SSH_HOST:?set SSH_HOST in config.env}"
: "${SSH_PORT:?set SSH_PORT in config.env}"
: "${SSH_USER:?set SSH_USER in config.env}"
: "${REMOTE_DIR:?set REMOTE_DIR in config.env}"

# SSH_KEY is optional. If unset/empty, fall back to ssh's default identities /
# ssh-agent (matching a plain `ssh -p PORT user@host` that works without -i).
SSH_KEY="${SSH_KEY:-}"
SSH_KEY="${SSH_KEY/#\~/$HOME}"

SSH_OPTS=(-p "$SSH_PORT"
          -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30)
if [ -n "$SSH_KEY" ]; then
  SSH_OPTS=(-i "$SSH_KEY" "${SSH_OPTS[@]}")
fi

remote_ssh() {
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$@"
}

remote_sh() {
  # Run a script passed on stdin on the remote host, with config env exported.
  remote_ssh "bash -s" "$@"
}

remote_rsync() {
  # Portable flags: macOS ships openrsync (2.6.9-compatible) which lacks
  # --info=progress2 and reliable -z. -a is archive; the link is fast enough.
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$@"
}

remote_target() {
  echo "${SSH_USER}@${SSH_HOST}"
}
