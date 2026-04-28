#!/usr/bin/env bash
set -euo pipefail

# gemma-mlx installer (macOS / Apple Silicon)
#  - creates ./venv
#  - installs mlx-lm (git main) + mlx-vlm
#  - generates ./bin/gemma and ./bin/gemma-photos wrappers
#  - adds aliases to ~/.zshrc (idempotent, marker-delimited block)

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

note()  { echo "${BOLD}==>${RESET} $*"; }
ok()    { echo "${GREEN}✓${RESET} $*"; }
warn()  { echo "${YELLOW}!${RESET} $*"; }
fail()  { echo "${RED}✗${RESET} $*" >&2; exit 1; }

# ---- preflight ----
[[ "$(uname -s)" == "Darwin" ]] || fail "This installer is macOS only."
[[ "$(uname -m)" == "arm64"  ]] || fail "Apple Silicon (M-series) required — Intel Macs are not supported."

PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || fail "python3 not found. Install via Xcode CLT (xcode-select --install) or Homebrew (brew install python)."

PYV="$($PY -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
PYV_MAJOR="${PYV%.*}"; PYV_MINOR="${PYV#*.}"
if (( PYV_MAJOR < 3 || (PYV_MAJOR == 3 && PYV_MINOR < 10) )); then
  fail "Python >= 3.10 required (found $PYV at $PY)."
fi
ok "Python $PYV at $PY"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/venv"
BINDIR="$REPO_DIR/bin"

# ---- venv ----
if [[ ! -x "$VENV/bin/python" ]]; then
  note "Creating virtual environment in $VENV"
  "$PY" -m venv "$VENV"
fi
ok "Virtualenv ready"

# ---- deps ----
note "Installing dependencies (a few minutes on first run)"
"$VENV/bin/pip" install --quiet --upgrade pip
# mlx-lm git main has the gemma4 model files; PyPI lags behind for new architectures.
# osxphotos: read Photos library directly so iCloud-only items can be analysed.
# pillow-heif: HEIC support for PIL (iPhone originals are HEIC by default).
# holidays: country-aware holiday calendars for the yearbook curator.
"$VENV/bin/pip" install --quiet --upgrade \
    "git+https://github.com/ml-explore/mlx-lm.git" \
    "mlx-vlm" \
    "osxphotos" \
    "pillow-heif" \
    "holidays" \
    "imagehash"
ok "Python dependencies installed"

# ---- wrappers ----
mkdir -p "$BINDIR"
cat > "$BINDIR/gemma" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" "$REPO_DIR/gemma.py" "\$@"
EOF
cat > "$BINDIR/gemma-photos" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" "$REPO_DIR/photos_caption.py" "\$@"
EOF
cat > "$BINDIR/gemma-yearbook" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" "$REPO_DIR/yearbook.py" "\$@"
EOF
chmod +x "$BINDIR/gemma" "$BINDIR/gemma-photos" "$BINDIR/gemma-yearbook"
ok "Wrappers written to $BINDIR"

# ---- shell aliases (zsh only — macOS default since Catalina) ----
ZSHRC="$HOME/.zshrc"
START="# >>> gemma-mlx (managed by install.sh) >>>"
END="# <<< gemma-mlx <<<"

# Strip any previous managed block before appending the new one.
if [[ -f "$ZSHRC" ]] && grep -qF "$START" "$ZSHRC"; then
  awk -v s="$START" -v e="$END" '
    $0==s {skip=1; next}
    $0==e {skip=0; next}
    !skip {print}
  ' "$ZSHRC" > "$ZSHRC.tmp" && mv "$ZSHRC.tmp" "$ZSHRC"
fi

cat >> "$ZSHRC" <<EOF

$START
alias gemma='$BINDIR/gemma'
alias gemma-photos='$BINDIR/gemma-photos'
alias gemma-yearbook='$BINDIR/gemma-yearbook'
$END
EOF
ok "Aliases added to ~/.zshrc"

echo
echo "${BOLD}${GREEN}Installation complete.${RESET}"
echo
echo "Open a new terminal (or run: ${BOLD}source ~/.zshrc${RESET}) and try:"
echo "  ${BOLD}gemma${RESET} 'hej, vem är du?'"
echo "  ${BOLD}gemma${RESET} -i path/to/photo.jpg 'beskriv vad du ser'"
echo "  ${BOLD}gemma-photos${RESET} --dry-run    # after selecting photos in Photos.app"
echo
echo "Or double-click ${BOLD}gemma-photos.command${RESET} / ${BOLD}gemma-yearbook.command${RESET} in Finder."
echo
echo "First run will download the model (~3.5 GB) from Hugging Face."
