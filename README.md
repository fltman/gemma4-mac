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

gemma-yearbook --year 2024 --count 100 --album "Året 2024"  # auto-curate a yearbook album
gemma-yearbook --year 2024 --dry-run                         # show what would be picked
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
./install.command
```

…or just **double-click `install.command` in Finder**. The `.command`
extension makes the same script Finder-runnable; no terminal required.

The installer:

1. Verifies Apple Silicon + Python ≥ 3.10
2. Creates `./venv` and installs Python deps: `mlx-lm` (from git main —
   PyPI lags behind on new architectures), `mlx-vlm`, `osxphotos`,
   `pillow-heif`, `holidays`, `imagehash`
3. Generates `bin/gemma`, `bin/gemma-photos` and `bin/gemma-yearbook`
   wrapper scripts
4. Adds three aliases to `~/.zshrc` inside an idempotent
   `# >>> gemma-mlx >>>` block

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
3. Either **double-click `gemma-photos.command`** or run from a terminal:

```bash
gemma-photos --dry-run             # see what it would write, without touching anything
gemma-photos                        # set caption + merge keywords
gemma-photos --no-caption           # only keywords
gemma-photos --no-keywords          # only caption
gemma-photos --replace-keywords     # overwrite existing keywords (default merges)
gemma-photos --replace-caption      # ignore the existing caption (default uses it as a hint)
gemma-photos --explicit-context     # weave date, place, and named people into the caption
gemma-photos --no-context           # ignore Photos metadata entirely
gemma-photos --style "poetic, two-line haiku"
gemma-photos --prompt "FULL CUSTOM PROMPT — must still emit CAPTION: and KEYWORDS: lines"
```

By default the photo's existing caption is fed to the model as a hint —
useful when humans got a detail right that the vision model gets wrong
(e.g. specific car brands). Use `--replace-caption` for a clean re-pass.

## `gemma-yearbook` — auto-curate a year in photos

Picks a balanced, deduplicated selection of photos from a date range and
creates a new album in Photos.app. The selection uses **Apple's own
per-photo aesthetic scores** (the same ones that drive the "Memories"
feature, read via osxphotos) for ranking — no extra ML pass needed.

Either **double-click `gemma-yearbook.command`** (uses defaults — 100 picks
from the current year) or run from a terminal:

```bash
gemma-yearbook --year 2024                                   # default: 100 photos, album "Yearbook 2024"
gemma-yearbook --year 2024 --count 50 --album "Best of '24"
gemma-yearbook --from 2024-06-01 --to 2024-08-31              # custom date range
gemma-yearbook --year 2024 --dry-run                          # report only, don't touch Photos
gemma-yearbook --year 2024 --holidays se,us                   # include US holidays too
gemma-yearbook --year 2024 --holidays none                    # ignore the holiday calendar
```

### How the selection works

The pipeline runs in this fixed order, narrowing ~2,000 candidate photos
down to the requested `--count`:

**1. Hard filters** (drops ~60–70% of the raw input)
Default exclusions: videos, screenshots, photos lacking EXIF camera info,
and photos without geolocation. Each can be opted back in
(`--include-videos`, `--include-screenshots`, `--include-no-camera`,
`--include-no-gps`).

**2. Bucketing** — every remaining photo lands in exactly one bucket:
- **Trips** — ≥`min_trip_size` photos taken ≥50 km from the year's
  geographic median, separated from other photo activity by ≥24h, **and
  containing at least `min_trip_persons` distinct tagged faces** across
  the whole trip. The face requirement is what separates a family
  vacation from a hospital visit, work conference or solo errand.
- **Holidays** — date matches the country's holiday calendar (via the
  Python `holidays` library), plus the *day before* so eves like
  Julafton, Midsommarafton and Nyårsafton are captured. Generic Sunday
  labels are filtered out.
- **Events** — dense time clusters: ≥8 photos within a ~4-hour window,
  total span ≤18h.
- **Everyday** — the long tail.

**3. Budget split** — defaults to 35% trips, 25% events, 20% holidays,
20% everyday. Empty buckets redistribute to the rest.

**4. Per-bucket selection** — different strategy per bucket type:
- **Trips** are weighted by `size × min(named_faces, 6)`. The trip budget
  is divided proportionally to that weight, so a 40-photo family trip
  ranks above a 40-photo solo conference. Within each trip, picks are
  spread proportionally across the trip's distinct dates so a 5-day trip
  doesn't collapse to its single most photogenic afternoon. Capped at
  `--max-per-trip` (default 20).
- **Holidays** distribute proportionally across distinct holidays,
  capped at `--max-per-cluster` (default 6).
- **Events** are taken from the largest clusters first, capped at
  `--max-per-cluster`.
- **Everyday** is straight top-N by quality.

  Within each bucket, candidates are ranked by quality:
  `score.overall − score.failure + 0.5·curation + 0.2·well_framed_subject
  + 0.1·sharply_focused_subject`.

**5. Dedup, in two passes** —
- **Scene dedup** (`--keep-per-scene`, default 2): groups photos by
  `(date, ~1km area)` and keeps only the top-N by quality from each
  group. This is the main weapon against passport-photo sessions and
  similar dense scenes — pHash is too coarse for "same wall, different
  pose".
- **pHash dedup** (`--similarity-threshold`, default 14): perceptual
  hash skips near-identical compositions that landed in different
  buckets. Secondary safety net.

**6. Topup to budget** — if scene dedup undershot the count, fills back
up from the year's highest-quality remaining photos, respecting both the
scene cap and the pHash threshold so duplicates aren't reintroduced.

**7. Trim to budget** — if the count is still over, drops the lowest-
quality photos.

**8. Person balance** (`--person-balance`, default 0.40) — if any tagged
person appears in more than 40% of the selection, swap their lowest-
quality photos for the year's best photos that don't include them
(dedup-aware). Set to 0 to disable.

**9. Album** — creates a new top-level album in Photos.app and adds the
picks via AppleScript.

The discovery report prints up-front, so `--dry-run` lets you sanity-
check the buckets and budget before committing to writes.

### Yearbook flag reference

| Flag | Default | Effect |
| --- | --- | --- |
| `--count` | 100 | Number of photos in the album |
| `--album` | `Yearbook YYYY` | Album name |
| `--holidays` | `se` | Country codes for holiday detection (e.g. `se,us`), or `none` |
| `--keep-per-scene` | 2 | Max per (date, ~1km area). Lower = stricter |
| `--similarity-threshold` | 14 | pHash distance cutoff (lower = stricter dedup) |
| `--max-per-cluster` | 6 | Cap per single event/holiday |
| `--max-per-trip` | 20 | Cap per single trip (trips get more headroom than events) |
| `--min-trip-size` | 8 | Photos required to qualify as a trip |
| `--min-trip-persons` | 2 | Distinct tagged faces required to qualify as a trip |
| `--person-balance` | 0.40 | Max share of selection any one person may appear in. 0 disables |
| `--include-videos`, `--include-screenshots`, `--include-no-camera`, `--include-no-gps` | off | Re-enable filtered content |
| `--dry-run` | off | Print plan without creating an album |

### Photos metadata as context

By default `gemma-photos` reads each photo's date, GPS-derived place name, and
any **named** faces from your Photos library and feeds them into the prompt as
soft context. The model uses them as tone hints — e.g. it'll prefer the
keyword `park` over `leaves` if the photo was taken in a park.

`--explicit-context` flips this from soft to hard: the model is told to weave
the place, date, and tagged person names directly into the caption text. The
difference for the same photo:

> **default:** Glad man leker med en ring bland grönska utomhus.
> **--explicit-context:** Anders ler vid Sofiero Park i Helsingborg under en solig eftermiddag i maj 2023.

Person names only get used if you've actually labelled the faces in Photos
(otherwise osxphotos returns `_UNKNOWN_`, which we filter out). `--no-context`
disables the whole thing.

The first run triggers a macOS dialog asking for permission to control Photos —
approve it.

**Tip:** don't change the Photos selection while the script is running. Lookup
falls back to a library-wide search by id if the selection has changed, but
keeping it stable is safer.

### Image source: previews, not originals

`gemma-photos` reads the **local preview derivative** straight from the
Photos library (via [osxphotos](https://github.com/RhetTbull/osxphotos))
rather than asking Photos.app to export the original. Three reasons:

1. **iCloud-only photos work.** With *Optimise Mac Storage* enabled, most
   originals live in iCloud and aren't on disk — but the previews are. So
   we can analyse cloud-only items without forcing slow downloads.
2. **iMessage attachments work.** Photos shared via iMessage live as their
   own AppleScript class (`«IPmi»`) and refuse to export through the normal
   API. Their preview derivatives, however, sit in
   `Photos Library/scopes/syndication/resources/derivatives/` and load fine
   — so they're handled the same way as any other photo.
3. **Originals add no value here.** Gemma's vision encoder resizes to ~768px
   internally, so the difference between a 4032×3024 HEIC original and an
   1080×1920 preview vanishes after preprocessing.

Preview sizes are typically 720–1080px on the long edge for recent photos,
sometimes as small as 480×360 for older library items. That's plenty for
scene captions and keywords; for pixel-level detail you'd need to read the
originals separately.

If reading the Photos library fails with a permission error, grant **Full
Disk Access** to your terminal: System Settings → Privacy & Security → Full
Disk Access → add Terminal (or iTerm).

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

Double-click `uninstall.command` in Finder, or:

```bash
./uninstall.command
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
