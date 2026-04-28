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
import re
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


TMP_EXPORT_DIR = Path("/tmp/gemma-photos-export")
IMAGE_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".tiff", ".tif"}


_ID_IN_ERROR = re.compile(r'id "([A-Fa-f0-9]{8}-[A-Fa-f0-9-]+(?:/L\d+/\d+)?)"')


def get_selection_items() -> list[dict]:
    """Resolve the current Photos selection.

    For each selected item returns a dict with keys
        id: str | None — Photos uuid (e.g. "UUID/L0/001")
        selection_index: int — 1-based index in the current selection

    For normal items `id of p` coerces fine. For iMessage attachments
    (AppleScript class «IPmi») coercion raises -1700 — but the error
    message AppleScript hands back contains the id literal, so we
    capture it and parse the uuid from there. Net effect: every
    selectable item ends up with an id and can be looked up via
    osxphotos.
    """
    script = '''
    tell application "Photos"
        set sel to (get selection)
        set total to count of sel
        set out to (total as text) & linefeed
        repeat with i from 1 to total
            try
                set theId to (id of item i of sel) as text
            on error errMsg
                set theId to "ERR:" & errMsg
            end try
            if theId is "" then set theId to "_NOID_"
            set out to out & theId & linefeed
        end repeat
        return out
    end tell
    '''
    lines = [line for line in osa(script).split("\n") if line.strip()]
    if not lines:
        return []
    total = int(lines[0])
    raw_ids = lines[1:]
    items = []
    for idx, raw in enumerate(raw_ids, 1):
        photo_id: str | None
        if raw.startswith("ERR:"):
            # Coercion errors come back with the *whole* selection list
            # described literally — `... id of item N of {«class IPmi»
            # id "UUID1" ..., «class IPmi» id "UUID2" ...} till typ text.`
            # — so we have to pick the i-th id, not the first match.
            all_ids = _ID_IN_ERROR.findall(raw[4:])
            photo_id = all_ids[idx - 1] if 1 <= idx <= len(all_ids) else None
        elif raw == "_NOID_":
            photo_id = None
        else:
            photo_id = raw
        items.append({"id": photo_id, "selection_index": idx})
    return items


def export_selection_items(indices: list[int]) -> dict[int, Path]:
    """Export each given selection index to its own subdirectory.

    Per-item with try-wrapper around the export — Photos refuses to
    resolve IPmi class items as `media item id "…"`, so those silently
    fail without taking down the whole batch. Returns only the indices
    that produced an image file; missing ones likely couldn't be
    exported via AppleScript and need another path (or aren't recoverable).
    """
    if TMP_EXPORT_DIR.exists():
        for entry in TMP_EXPORT_DIR.iterdir():
            if entry.is_file():
                entry.unlink()
            else:
                for f in entry.iterdir():
                    if f.is_file():
                        f.unlink()
                entry.rmdir()
    TMP_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths: dict[int, Path] = {}
    for sidx in indices:
        item_dir = TMP_EXPORT_DIR / str(sidx)
        item_dir.mkdir(parents=True, exist_ok=True)
        script = f'''
        tell application "Photos"
            set sel to (get selection)
            try
                export {{item {sidx} of sel}} to (POSIX file "{item_dir}" as alias) without using originals
            end try
        end tell
        '''
        osa(script)
        files = sorted(
            f for f in item_dir.glob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        if files:
            paths[sidx] = files[0]
    return paths


def find_preview_path(db, photo_id: str) -> Path | None:
    """Return an on-disk preview derivative for a Photos id, or None.

    Only returns derivative previews — never originals. Gemma's vision
    encoder resizes to ~768 internally, so loading a 50MB HEIC original
    just slows things down. When no derivative is on disk (e.g. iCloud-
    only with derivative purged), the caller falls back to bulk export.
    """
    uuid = photo_id.split("/")[0]
    photo = db.get_photo(uuid)
    if photo is None:
        return None
    for d in photo.path_derivatives or []:
        if d and Path(d).exists():
            return Path(d)
    return None


def resolve_preview_paths(items: list[dict], db) -> dict[int, Path]:
    """Resolve a preview JPEG/HEIC path for each selection item.

    Fast path: on-disk derivative via osxphotos. iMessage items live
    under `scopes/syndication/resources/derivatives/` and osxphotos
    finds them just fine — provided we managed to extract the id from
    the AppleScript error in `get_selection_items`.

    Slow path (only when the derivative isn't on disk, e.g. iCloud-only):
    per-item AppleScript export with try-wrapper, so IPmi items that
    Photos refuses to export silently drop out without taking the
    rest of the batch with them.
    """
    paths: dict[int, Path] = {}
    needs_export: list[dict] = []
    for item in items:
        if item["id"]:
            preview = find_preview_path(db, item["id"])
            if preview:
                paths[item["selection_index"]] = preview
                continue
        needs_export.append(item)

    if not needs_export:
        return paths

    print("Exporterar förhandsvisningar via Photos…", end=" ", flush=True)
    indices = [it["selection_index"] for it in needs_export]
    exported = export_selection_items(indices)
    print(f"{len(exported)}/{len(indices)} fil(er).")
    paths.update(exported)
    return paths


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
    existing_caption: str | None = None,
) -> tuple[str, list[str]]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    parts = []
    if context and explicit_context:
        parts.append(
            "Kontext för bilden:\n"
            f"{context}\n\n"
            "Väv in plats, datum (eller månad/år) och eventuella personnamn "
            "från kontexten naturligt i CAPTION. Använd ENDAST de namn som "
            "är listade ovan — beskriv andra personer anonymt."
        )
    elif context:
        parts.append(
            "Kontext för bilden (använd endast om relevant för det du ser):\n"
            f"{context}"
        )
    if existing_caption:
        parts.append(
            "Befintlig caption (använd som vägledning vid tveksamma detaljer "
            "— t.ex. specifika bilmärken, platser eller namn som är svåra att "
            "se i bilden — men korrigera tydliga fel):\n"
            f"{existing_caption}"
        )
    parts.append(prompt)
    full_prompt = "\n\n".join(parts)
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
    ap.add_argument("--replace-caption", action="store_true",
                    help="Ignore the existing caption when generating a new "
                         "one. Default: pass it to the model as a hint to "
                         "help disambiguate details (e.g. specific car brands).")
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

    items = get_selection_items()
    if not items:
        print("Inget valt i Photos. Markera bilder och kör igen.")
        return
    print(f"Bearbetar {len(items)} bild(er){' (dry-run)' if args.dry_run else ''}…\n")

    print("Läser Photos-bibliotek…", end=" ", flush=True)
    import osxphotos
    db = osxphotos.PhotosDB()
    print(f"{len(db.photos())} bilder.")

    paths = resolve_preview_paths(items, db)

    print("Laddar Gemma…", end=" ", flush=True)
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(REPO)
    config = load_config(REPO)
    print("klart.\n")

    for i, item in enumerate(items, 1):
        pid = item["id"]
        try:
            img = paths.get(item["selection_index"])
            if img is None:
                print(f"  [{i}/{len(items)}] ✗ Ingen bildfil hittades")
                continue

            photo = db.get_photo(pid.split("/")[0]) if pid else None
            context = None
            if not args.no_context and photo:
                context = build_context_block(photo)
            existing_caption = None
            if not args.replace_caption and photo and photo.description:
                existing_caption = photo.description.strip() or None
            caption, keywords = analyze_image(
                model, processor, config, img, prompt,
                context=context, explicit_context=args.explicit_context,
                existing_caption=existing_caption,
            )

            lines = [f"  [{i}/{len(items)}]"]
            if not pid:
                lines.append("(saknar id i Photos — analyserar men kan ej "
                             "skriva tillbaka caption/nyckelord)")
            if not args.no_caption and caption:
                if not args.dry_run and pid:
                    set_description(pid, caption)
                lines.append(f"caption: {caption}")
            if not args.no_keywords and keywords:
                if not args.dry_run and pid:
                    set_keywords(pid, keywords, merge=not args.replace_keywords)
                lines.append(f"keywords: {', '.join(keywords)}")
            if not caption and not keywords:
                lines.append("(modellen svarade utan caption/keywords)")
            print("\n    ".join(lines))
        except Exception as e:
            print(f"  [{i}/{len(items)}] ✗ {e}")

    print("\nKlart.")


if __name__ == "__main__":
    main()
