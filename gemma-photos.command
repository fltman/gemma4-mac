#!/usr/bin/env bash
set -euo pipefail

# Double-clickable launcher for gemma-photos.
# Works from Finder (double-click) and from a terminal.
# Expects install.command to have created bin/gemma-photos beforehand.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$REPO_DIR/bin/gemma-photos"

if [[ ! -x "$WRAPPER" ]]; then
    echo "Hittar inte $WRAPPER — kör install.command först."
    exit 1
fi

cd "$REPO_DIR"
"$WRAPPER" "$@"
