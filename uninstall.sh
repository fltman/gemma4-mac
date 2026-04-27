#!/usr/bin/env bash
set -euo pipefail

# gemma-mlx uninstaller — removes aliases, optionally venv + wrappers.

RED=$'\033[31m'; GREEN=$'\033[32m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZSHRC="$HOME/.zshrc"
START="# >>> gemma-mlx (managed by install.sh) >>>"
END="# <<< gemma-mlx <<<"

if [[ -f "$ZSHRC" ]] && grep -qF "$START" "$ZSHRC"; then
  awk -v s="$START" -v e="$END" '
    $0==s {skip=1; next}
    $0==e {skip=0; next}
    !skip {print}
  ' "$ZSHRC" > "$ZSHRC.tmp" && mv "$ZSHRC.tmp" "$ZSHRC"
  echo "${GREEN}✓${RESET} Removed aliases from ~/.zshrc"
else
  echo "No managed alias block found in ~/.zshrc."
fi

read -r -p "Also remove venv and bin/ in $REPO_DIR? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  rm -rf "$REPO_DIR/venv" "$REPO_DIR/bin"
  echo "${GREEN}✓${RESET} Removed venv and bin/"
fi

echo
echo "${BOLD}Done.${RESET} Restart your shell (or run: source ~/.zshrc)."
echo "Note: the Hugging Face model cache is untouched — clear with:"
echo "  rm -rf ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit"
