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


def build_context_block(photo) -> str | None:
    """Compose a 'Kontext'-block from Photos metadata, or return None.

    Includes date/time, reverse-geocoded place name, and tagged person names.
    Untagged faces (osxphotos returns '_UNKNOWN_') are skipped — they'd just
    confuse the model.
    """
    parts = []
    if photo.date:
        parts.append(f"Datum: {photo.date.strftime('%Y-%m-%d %H:%M')}")
    place = photo.place
    if place and getattr(place, "name", None):
        parts.append(f"Plats: {place.name}")
    if photo.persons:
        named = [p for p in photo.persons if p and not p.startswith("_")]
        if named:
            parts.append(f"Personer i bilden: {', '.join(named)}")
    return "\n".join(parts) if parts else None


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


def analyze_image(
    model, processor, config, image_path: Path, prompt: str,
    context: str | None = None, explicit_context: bool = False,
) -> tuple[str, list[str]]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    if context and explicit_context:
        full_prompt = (
            "Kontext för bilden:\n"
            f"{context}\n\n"
            "Väv in plats, datum (eller månad/år) och eventuella personnamn "
            "från kontexten naturligt i CAPTION. Använd ENDAST de namn som "
            "är listade ovan — beskriv andra personer anonymt.\n\n"
            f"{prompt}"
        )
    elif context:
        full_prompt = (
            "Kontext för bilden (använd endast om relevant för det du ser):\n"
            f"{context}\n\n{prompt}"
        )
    else:
        full_prompt = prompt
    formatted = apply_chat_template(processor, config, full_prompt, num_images=1)
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
    ap.add_argument("--no-context", action="store_true",
                    help="Don't inject Photos metadata (date, place, named "
                         "people) into the prompt. Default: include it.")
    ap.add_argument("--explicit-context", action="store_true",
                    help="Force the model to weave date, place, and named "
                         "people into the caption text (not just use them as "
                         "tone hints). Implies context is on.")
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
            context = None
            if not args.no_context:
                photo = db.get_photo(pid.split("/")[0])
                if photo:
                    context = build_context_block(photo)
            caption, keywords = analyze_image(
                model, processor, config, img, prompt,
                context=context, explicit_context=args.explicit_context,
            )

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
