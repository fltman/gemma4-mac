#!/usr/bin/env python3
"""Caption + tag selected photos in Apple Photos using local Gemma 4 e4b.

Workflow:
  1. Open Photos.app, select one or more photos.
  2. Run this script.
  3. For each selection it exports the photo, asks Gemma for a Swedish
     caption + 3-7 keywords, and writes both back to Photos.

Flags:
  --dry-run            Print results but don't write back to Photos.
  --no-caption         Skip caption (only set keywords).
  --no-keywords        Skip keywords (only set caption).
  --replace-keywords   Overwrite existing keywords. Default merges (preserves
                       any keywords you've added manually).
"""
import argparse
import subprocess
from pathlib import Path

# Register HEIF/HEIC support so PIL (used by mlx-vlm) can read iPhone originals.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

REPO = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_STYLE = "kort beskrivande mening på svenska, max ~15 ord"


def build_prompt(style: str) -> str:
    return (
        "Analysera bilden och svara EXAKT i detta format, inget annat:\n"
        f"CAPTION: <{style}>\n"
        "KEYWORDS: <3-7 relevanta nyckelord på svenska, kommaseparerade, gemener>"
    )


def osa(script: str) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"AppleScript failed: {r.stderr.strip()}")
    return r.stdout.strip()


def applescript_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def get_selection_ids() -> list[str]:
    script = '''
    tell application "Photos"
        set sel to (get selection)
        set out to ""
        repeat with p in sel
            set out to out & (id of p) & linefeed
        end repeat
        return out
    end tell
    '''
    return [line for line in osa(script).split("\n") if line]


# AppleScript snippet that resolves `targetId` (already declared) to a media
# item bound to `p`. Avoids `media item id "..."` direct lookup, which can
# misbehave when ids contain slashes on some macOS versions.
_RESOLVE = '''
        set p to missing value
        try
            set p to first item of ((get selection) whose id is targetId)
        end try
        if p is missing value then
            try
                set p to first item of (every media item whose id is targetId)
            end try
        end if
        if p is missing value then error "media item not found: " & targetId
'''


def find_local_path(db, photo_id: str) -> Path:
    """Resolve a Photos selection id to a locally-available image.

    Prefer the largest preview derivative — always JPEG, present even for
    iCloud-only photos, and uniformly sized (Gemma's vision encoder resizes
    to ~768 internally so original-resolution gains nothing). Falls back to
    edited/original masters only if no derivative exists.
    """
    uuid = photo_id.split("/")[0]  # strip "/L0/001" suffix
    photo = db.get_photo(uuid)
    if photo is None:
        raise RuntimeError(f"photo not found in library: {uuid}")
    for d in photo.path_derivatives or []:
        if d and Path(d).exists():
            return Path(d)
    for candidate in (photo.path_edited, photo.path):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise RuntimeError(
        f"no local image data for {uuid} — derivative may have been purged"
    )


def set_description(photo_id: str, text: str) -> None:
    script = f'''
    tell application "Photos"
        set targetId to "{applescript_string(photo_id)}"
        {_RESOLVE}
        set description of p to "{applescript_string(text)}"
    end tell
    '''
    osa(script)


def set_keywords(photo_id: str, keywords: list[str], merge: bool) -> None:
    if not keywords:
        return
    kw_literal = "{" + ", ".join(f'"{applescript_string(k)}"' for k in keywords) + "}"
    safe_id = applescript_string(photo_id)
    if merge:
        body = f'''
            set existing to keywords of p
            if existing is missing value then set existing to {{}}
            set merged to existing
            set toAdd to {kw_literal}
            repeat with kw in toAdd
                set kwStr to kw as string
                if merged does not contain kwStr then
                    set merged to merged & {{kwStr}}
                end if
            end repeat
            set keywords of p to merged
        '''
    else:
        body = f"set keywords of p to {kw_literal}"
    script = f'''
    tell application "Photos"
        set targetId to "{safe_id}"
        {_RESOLVE}
        {body}
    end tell
    '''
    osa(script)


def parse_response(text: str) -> tuple[str, list[str]]:
    caption, keywords = "", []
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CAPTION:"):
            caption = line.split(":", 1)[1].strip()
        elif line.upper().startswith("KEYWORDS:"):
            raw = line.split(":", 1)[1].strip()
            keywords = [k.strip().lower() for k in raw.split(",") if k.strip()]
    return caption, keywords


def analyze_image(model, processor, config, image_path: Path, prompt: str) -> tuple[str, list[str]]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    result = generate(
        model, processor, formatted,
        image=[str(image_path)],
        max_tokens=200, verbose=False,
    )
    text = result.text if hasattr(result, "text") else str(result)
    return parse_response(text.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print results but don't write back to Photos")
    ap.add_argument("--no-caption", action="store_true",
                    help="Skip caption (only set keywords)")
    ap.add_argument("--no-keywords", action="store_true",
                    help="Skip keywords (only set caption)")
    ap.add_argument("--replace-keywords", action="store_true",
                    help="Overwrite existing keywords (default: merge)")
    ap.add_argument("--style", default=DEFAULT_STYLE,
                    help="Tone/style for the caption (e.g. 'poetisk, två rader')")
    ap.add_argument("--prompt", default=None,
                    help="Full prompt override. Must still elicit "
                         "'CAPTION:' and 'KEYWORDS:' lines.")
    args = ap.parse_args()

    prompt = args.prompt if args.prompt else build_prompt(args.style)

    if args.no_caption and args.no_keywords:
        print("--no-caption + --no-keywords → inget att göra.")
        return

    ids = get_selection_ids()
    if not ids:
        print("Inget valt i Photos. Markera bilder och kör igen.")
        return
    print(f"Bearbetar {len(ids)} bild(er){' (dry-run)' if args.dry_run else ''}…\n")

    print("Läser Photos-bibliotek…", end=" ", flush=True)
    import osxphotos
    db = osxphotos.PhotosDB()
    print(f"{len(db.photos())} bilder.")

    print("Laddar Gemma…", end=" ", flush=True)
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(REPO)
    config = load_config(REPO)
    print("klart.\n")

    for i, pid in enumerate(ids, 1):
        try:
            img = find_local_path(db, pid)
            caption, keywords = analyze_image(model, processor, config, img, prompt)

            lines = [f"  [{i}/{len(ids)}]"]
            if not args.no_caption and caption:
                if not args.dry_run:
                    set_description(pid, caption)
                lines.append(f"caption: {caption}")
            if not args.no_keywords and keywords:
                if not args.dry_run:
                    set_keywords(pid, keywords, merge=not args.replace_keywords)
                lines.append(f"keywords: {', '.join(keywords)}")
            if len(lines) == 1:
                lines.append("(modellen svarade utan caption/keywords)")
            print("\n    ".join(lines))
        except Exception as e:
            print(f"  [{i}/{len(ids)}] ✗ {e}")

    print("\nKlart.")


if __name__ == "__main__":
    main()
