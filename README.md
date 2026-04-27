# gemma4-mac

Run **Gemma 4 e4b** locally on Apple Silicon — text, vision, and an Apple Photos
auto-captioner that writes results back into your library.

Built on [mlx-lm](https://github.com/ml-explore/mlx-lm) and
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm). All inference is on-device.

## What you get

```bash
gemma "skriv en haiku om Apple Silicon"             # text chat
gemma                                                # interactive REPL
gemma -i sunset.jpg "describe this in one sentence"  # vision
gemma -a clip.mp3 "transcribe the audio"             # audio

gemma-photos --dry-run                # caption + tag the current Photos.app selection
gemma-photos --style "poetic, two lines"
```

`gemma-photos` exports each selected photo, asks Gemma for a Swedish caption +
3–7 keywords, and writes both back into the photo's metadata in Photos. Default
keyword behaviour merges with existing tags (no duplicates, nothing lost).

## Requirements

- **macOS on Apple Silicon** (M1 / M2 / M3 / M4 / M5). Intel Macs are not
  supported — Gemma 4 e4b uses MLX which is Metal-only.
- **Python ≥ 3.10** (3.13/3.14 from Homebrew works; the system Python in Xcode
  CLT is fine too).
- **~5 GB free disk** for the 4-bit quantised model weights.
- **16 GB RAM** is plenty. Peak inference is ~5.8 GB (vision encoder + LLM).
- For Photos integration: **Photos.app** and one-time approval to let Terminal
  control it (macOS prompts the first time).

## Install

```bash
git clone https://github.com/fltman/gemma4-mac.git
cd gemma4-mac
./install.sh
```

The installer:

1. Verifies Apple Silicon + Python ≥ 3.10
2. Creates `./venv` and installs `mlx-lm` (from git main; PyPI lags behind on
   new architectures) and `mlx-vlm`
3. Generates `bin/gemma` and `bin/gemma-photos` wrapper scripts
4. Adds two aliases to `~/.zshrc` inside an idempotent `# >>> gemma-mlx >>>`
   block

Open a new terminal (or `source ~/.zshrc`) and you're done.

The first call will download the model (~3.5 GB) from Hugging Face into
`~/.cache/huggingface/hub/`.

## Usage

### `gemma` — text + multimodal CLI

```bash
gemma "förklara MLX för någon som kan PyTorch"      # one-shot
gemma                                                # interactive (REPL)
gemma -i a.jpg -i b.jpg "compare these two photos"   # multiple images
gemma -a interview.m4a "summarise this in 3 bullets" # audio in
```

In interactive mode: `/reset` clears history, `Ctrl-D` exits.

### `gemma-photos` — Apple Photos auto-captioner

1. Open Photos.app
2. Select one or more photos (Cmd-click for multi-select)
3. Run:

```bash
gemma-photos --dry-run        # see what it would write, without touching anything
gemma-photos                   # set caption + merge keywords
gemma-photos --no-caption      # only keywords
gemma-photos --no-keywords     # only caption
gemma-photos --replace-keywords  # overwrite existing keywords (default merges)
gemma-photos --style "poetic, two-line haiku"
gemma-photos --prompt "FULL CUSTOM PROMPT — must still emit CAPTION: and KEYWORDS: lines"
```

The first run triggers a macOS dialog asking for permission to control Photos —
approve it.

**Tip:** don't change the Photos selection while the script is running. Lookup
falls back to a library-wide search by id if the selection has changed, but
keeping it stable is safer.

## Performance

Measured on a MacBook Pro M5 (10-core, 16 GB RAM):

| Workload                           | Speed             |
| ---------------------------------- | ----------------- |
| Text generation                    | ~28 tokens/sec    |
| Image prompt prefill (1 image)     | ~57 tokens/sec    |
| Peak memory (text only)            | 4.3 GB            |
| Peak memory (vision)               | 5.8 GB            |
| Cold model load                    | ~6 sec            |

Per-photo cost in `gemma-photos` is roughly **5–8 seconds** end-to-end
(export + prefill + 5–7 keywords + caption).

## Customisation

The Photos prompt template is in `photos_caption.py` (`build_prompt`). The
default asks for a Swedish caption, ≤15 words, plus 3–7 lowercase Swedish
keywords. Use `--style` for tone tweaks or `--prompt` to replace it entirely.

To run a different Gemma variant, change `REPO` at the top of `gemma.py` /
`photos_caption.py` to e.g. `mlx-community/gemma-4-e4b-it-8bit` (better
quality, more RAM) or `mlx-community/gemma-4-e2b-it-4bit` (smaller, faster).

## Uninstall

```bash
./uninstall.sh
```

Removes the alias block from `~/.zshrc` and (with confirmation) the local
`venv/` and `bin/`. The Hugging Face model cache is left alone — the script
prints the exact path if you want to nuke it.

## Why not Ollama?

Ollama works fine for text and is a one-liner (`ollama run gemma4:e4b`). MLX is
worth the extra setup when:

- You want vision/audio (Ollama support varies by model)
- You're on Apple Silicon and want the Metal-native path
- You want to script around the Python API (e.g. the Photos integration here)

For straight terminal chat, Ollama is honestly easier. This repo is built for
the cases above.

## Support

If this is useful and you'd like to support more work like it:

- ☕ **Patreon** — [patreon.com/AndersBjarby](https://www.patreon.com/AndersBjarby)
- 📘 **Book** — [The Evolution of AI Agentic Thinking](https://anders.bjarby.com/the-evolution-of-ai-agentic-thinking/)

## License

MIT — see `LICENSE`.

## Acknowledgements

- [Apple MLX](https://github.com/ml-explore/mlx) — the framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — text generation
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — vision/audio generation
- [mlx-community](https://huggingface.co/mlx-community) — quantised conversions
- Google DeepMind — the Gemma 4 model
