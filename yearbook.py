#!/usr/bin/env python3
"""Yearbook curator for Apple Photos.

Picks a balanced selection of photos from a time range and drops them into a
new album. Uses Apple's own per-photo aesthetic scores (exposed via osxphotos)
to rank candidates, so we don't need to run a vision model on thousands of
images. Buckets photos into trips, holidays, event clusters, and everyday
gems, then allocates a budget across them and dedupes near-duplicates.

Usage:
  yearbook.py --year 2024 --count 100 --album "Året 2024"
  yearbook.py --year 2024 --dry-run        # show plan, don't create album
  yearbook.py --from 2024-06-01 --to 2024-08-31 --count 50
"""
from __future__ import annotations

import argparse
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from math import radians, sin, cos, sqrt, atan2
from typing import Iterable

import osxphotos
import holidays as pyholidays


# --------------------------------------------------------------------------
# Date range
# --------------------------------------------------------------------------

def parse_date_range(args) -> tuple[datetime, datetime]:
    if args.year:
        return (datetime(args.year, 1, 1),
                datetime(args.year, 12, 31, 23, 59, 59))
    if args.from_ and args.to:
        return (datetime.fromisoformat(args.from_),
                datetime.fromisoformat(args.to + " 23:59:59"))
    raise SystemExit("Specify --year YYYY or both --from YYYY-MM-DD --to YYYY-MM-DD")


# --------------------------------------------------------------------------
# Geo helpers
# --------------------------------------------------------------------------

def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    h = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))


def home_centroid(photos: list) -> tuple[float, float] | None:
    locs = [p.location for p in photos
            if p.location and p.location[0] is not None and p.location[1] is not None]
    if not locs:
        return None
    locs.sort()
    return locs[len(locs) // 2]  # geographic median (cheap)


# --------------------------------------------------------------------------
# Holidays
# --------------------------------------------------------------------------

# Filter out generic weekly off-days (Sunday is legally a holiday in Sweden,
# but for yearbook purposes we only care about dated celebrations).
_GENERIC_HOLIDAY_NAMES = {"Söndag", "Sunday"}


def expand_holidays(country_codes: list[str], years: list[int]) -> dict[date, str]:
    """Return a dict mapping each holiday date (and the day before) to a label.

    The day before is included so that e.g. Dec 24 (Julafton) is captured
    even when only Dec 25 (Juldagen) is in the legal-holidays list.
    """
    out: dict[date, str] = {}
    for code in country_codes:
        try:
            cal = pyholidays.country_holidays(code.upper(), years=years)
        except Exception:
            print(f"  warning: unknown country code '{code}'")
            continue
        for d, raw_name in cal.items():
            # The library may concatenate multiple labels with "; " when a
            # date is both a public holiday and a Sunday — strip generic ones.
            parts = [p.strip() for p in raw_name.split(";")
                     if p.strip() and p.strip() not in _GENERIC_HOLIDAY_NAMES]
            if not parts:
                continue
            name = "; ".join(parts)
            out.setdefault(d, name)
            eve = d - timedelta(days=1)
            if eve not in out:
                out[eve] = f"{name} (eve)"
    return out


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

@dataclass
class Cluster:
    photos: list = field(default_factory=list)
    label: str = ""

    @property
    def start(self): return min(p.date for p in self.photos)
    @property
    def end(self): return max(p.date for p in self.photos)
    @property
    def duration_h(self): return (self.end - self.start).total_seconds() / 3600


def split_by_time_gap(photos: list, gap_hours: float = 6.0) -> list[list]:
    if not photos:
        return []
    photos = sorted(photos, key=lambda p: p.date)
    out, current = [], [photos[0]]
    for p in photos[1:]:
        if (p.date - current[-1].date).total_seconds() / 3600 < gap_hours:
            current.append(p)
        else:
            out.append(current)
            current = [p]
    out.append(current)
    return out


def detect_trips(photos: list, home: tuple[float, float] | None,
                 min_dist_km: float = 50.0) -> list[Cluster]:
    if not home:
        return []
    away = [p for p in photos
            if p.location
            and p.location[0] is not None and p.location[1] is not None
            and haversine_km(p.location, home) >= min_dist_km]
    trip_clusters = []
    for group in split_by_time_gap(away, gap_hours=24):
        if len(group) < 5:
            continue
        # Try to label with most common place name
        names = Counter(p.place.name for p in group if p.place and p.place.name)
        label = names.most_common(1)[0][0] if names else "Trip"
        # Truncate verbose place names
        if "," in label:
            label = label.rsplit(",", 1)[-1].strip() + " — " + label.split(",")[0].strip()
        trip_clusters.append(Cluster(photos=group, label=f"Trip: {label}"))
    return trip_clusters


def detect_event_clusters(photos: list,
                          min_size: int = 8,
                          max_h: float = 18.0) -> list[Cluster]:
    out = []
    for group in split_by_time_gap(photos, gap_hours=4):
        if len(group) < min_size:
            continue
        c = Cluster(photos=group)
        if c.duration_h > max_h:
            continue
        c.label = f"Event {c.start.date()}"
        out.append(c)
    return out


# --------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------

@dataclass
class Buckets:
    trips: list[Cluster] = field(default_factory=list)
    holidays: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    events: list[Cluster] = field(default_factory=list)
    everyday: list = field(default_factory=list)


def bucket_photos(photos: list, holiday_map: dict[date, str]) -> Buckets:
    home = home_centroid(photos)
    b = Buckets()
    b.trips = detect_trips(photos, home)
    trip_uuids = {p.uuid for c in b.trips for p in c.photos}

    non_trip = [p for p in photos if p.uuid not in trip_uuids]
    holiday_uuids = set()
    for p in non_trip:
        if p.date and p.date.date() in holiday_map:
            label = holiday_map[p.date.date()]
            b.holidays[label].append(p)
            holiday_uuids.add(p.uuid)

    rest = [p for p in non_trip if p.uuid not in holiday_uuids]
    b.events = detect_event_clusters(rest)
    event_uuids = {p.uuid for c in b.events for p in c.photos}

    b.everyday = [p for p in rest if p.uuid not in event_uuids]
    return b


# --------------------------------------------------------------------------
# Scoring + dedup
# --------------------------------------------------------------------------

def quality(p) -> float:
    s = p.score
    if s is None:
        return 0.0
    overall = (s.overall or 0)
    failure = (s.failure or 0)
    curation = (s.curation or 0)
    well_framed = (s.well_framed_subject or 0)
    sharply = (s.sharply_focused_subject or 0)
    return overall - failure + 0.5 * curation + 0.2 * well_framed + 0.1 * sharply


def dedup_burst(photos: list, window_seconds: int = 30) -> list:
    """Within each <window_seconds> burst, keep only the highest-quality photo."""
    if not photos:
        return []
    photos = sorted(photos, key=lambda p: p.date)
    kept = [photos[0]]
    for p in photos[1:]:
        if (p.date - kept[-1].date).total_seconds() < window_seconds:
            if quality(p) > quality(kept[-1]):
                kept[-1] = p
        else:
            kept.append(p)
    return kept


def select_top(photos: list, n: int) -> list:
    if n <= 0 or not photos:
        return []
    deduped = dedup_burst(photos)
    return sorted(deduped, key=quality, reverse=True)[:n]


# --------------------------------------------------------------------------
# Budget allocation
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {"trips": 0.35, "holidays": 0.20, "events": 0.25, "everyday": 0.20}


def allocate(buckets: Buckets, total: int,
             weights: dict = DEFAULT_WEIGHTS) -> dict[str, int]:
    available = {
        "trips": sum(len(c.photos) for c in buckets.trips),
        "holidays": sum(len(p) for p in buckets.holidays.values()),
        "events": sum(len(c.photos) for c in buckets.events),
        "everyday": len(buckets.everyday),
    }
    targets = {k: min(int(round(weights[k] * total)), available[k]) for k in weights}
    leftover = total - sum(targets.values())
    # Redistribute leftover into buckets that still have headroom (in priority order)
    while leftover > 0 and any(available[k] > targets[k] for k in available):
        for k in ("trips", "events", "holidays", "everyday"):
            if leftover <= 0:
                break
            if targets[k] < available[k]:
                targets[k] += 1
                leftover -= 1
    return targets


# --------------------------------------------------------------------------
# AppleScript: create album
# --------------------------------------------------------------------------

def applescript_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def create_album_with(name: str, uuids: Iterable[str]) -> int:
    """Create a new top-level album in Photos and add given UUIDs. Returns count added."""
    safe_name = applescript_string(name)
    uuid_list = "{" + ", ".join(f'"{applescript_string(u)}"' for u in uuids) + "}"
    script = f'''
    tell application "Photos"
        set newAlbum to make new album named "{safe_name}"
        set added to 0
        repeat with targetId in {uuid_list}
            set found to {{}}
            try
                set found to (every media item whose id is targetId)
            end try
            if (count of found) > 0 then
                add found to newAlbum
                set added to added + 1
            end if
        end repeat
        return added as string
    end tell
    '''
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"AppleScript failed: {r.stderr.strip()}")
    return int(r.stdout.strip() or 0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_discovery(photos: list, buckets: Buckets):
    print(f"Photos in range: {len(photos)}")

    if buckets.trips:
        print(f"\nTrips ({len(buckets.trips)}):")
        for c in buckets.trips:
            print(f"  {c.start.date()}–{c.end.date()}  {c.label} ({len(c.photos)} photos)")

    if buckets.holidays:
        print(f"\nHolidays ({sum(len(v) for v in buckets.holidays.values())} photos):")
        # Group by base label (strip "(eve)")
        by_label = defaultdict(int)
        for label, lst in buckets.holidays.items():
            base = label.replace(" (eve)", "")
            by_label[base] += len(lst)
        for label, count in sorted(by_label.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count} photos")

    if buckets.events:
        print(f"\nEvent clusters ({len(buckets.events)}):")
        for c in sorted(buckets.events, key=lambda c: -len(c.photos))[:10]:
            print(f"  {c.start.date()} ({c.start.strftime('%a')}) — {len(c.photos)} photos in {c.duration_h:.1f}h")
        if len(buckets.events) > 10:
            print(f"  ... and {len(buckets.events) - 10} more")

    print(f"\nEveryday gems pool: {len(buckets.everyday)} photos")

    # Top people
    persons = Counter()
    for p in photos:
        for name in (p.persons or []):
            if name and not name.startswith("_"):
                persons[name] += 1
    if persons:
        print(f"\nTop named people:")
        for name, count in persons.most_common(8):
            print(f"  {name}: {count}")


def print_plan(targets: dict[str, int], buckets: Buckets):
    print(f"\nBudget allocation:")
    print(f"  trips:    {targets['trips']:>3} photos (from {sum(len(c.photos) for c in buckets.trips)} available)")
    print(f"  holidays: {targets['holidays']:>3} photos (from {sum(len(p) for p in buckets.holidays.values())} available)")
    print(f"  events:   {targets['events']:>3} photos (from {sum(len(c.photos) for c in buckets.events)} available)")
    print(f"  everyday: {targets['everyday']:>3} photos (from {len(buckets.everyday)} available)")


# --------------------------------------------------------------------------
# Selection per bucket
# --------------------------------------------------------------------------

def pick_from_buckets(buckets: Buckets, targets: dict[str, int]) -> list:
    selected = []

    # trips: distribute across trips proportionally to size
    trips_target = targets["trips"]
    if buckets.trips and trips_target:
        sizes = [len(c.photos) for c in buckets.trips]
        total_trip = sum(sizes)
        for c, sz in zip(buckets.trips, sizes):
            n = max(1, round(trips_target * sz / total_trip))
            selected.extend(select_top(c.photos, n))

    # holidays: distribute proportionally across distinct holidays
    holiday_target = targets["holidays"]
    if buckets.holidays and holiday_target:
        # group by base name (drop eve marker)
        by_base = defaultdict(list)
        for label, lst in buckets.holidays.items():
            base = label.replace(" (eve)", "")
            by_base[base].extend(lst)
        sizes = {k: len(v) for k, v in by_base.items()}
        total_h = sum(sizes.values())
        for k, lst in by_base.items():
            n = max(1, round(holiday_target * sizes[k] / total_h))
            selected.extend(select_top(lst, n))

    # events: take from biggest clusters first
    event_target = targets["events"]
    if buckets.events and event_target:
        sorted_events = sorted(buckets.events, key=lambda c: -len(c.photos))
        per_event = max(1, event_target // max(1, min(len(sorted_events), 6)))
        remaining = event_target
        for c in sorted_events:
            if remaining <= 0:
                break
            n = min(per_event, remaining)
            selected.extend(select_top(c.photos, n))
            remaining -= n

    # everyday gems: top by quality across what's left
    everyday_target = targets["everyday"]
    if buckets.everyday and everyday_target:
        selected.extend(select_top(buckets.everyday, everyday_target))

    # Final dedup by uuid (a photo might have landed in two buckets in edge cases)
    seen = set()
    final = []
    for p in selected:
        if p.uuid not in seen:
            seen.add(p.uuid)
            final.append(p)
    return final


def trim_to_budget(selected: list, total: int) -> list:
    """Trim the selection to exactly `total` photos.

    Drops the lowest-scoring entries while preserving bucket order — trips
    and holidays were picked intentionally and shouldn't be sacrificed before
    everyday filler.
    """
    if len(selected) <= total:
        return selected
    excess = len(selected) - total
    # Find indices of `excess` lowest-scoring photos and remove them.
    indices_by_q = sorted(range(len(selected)), key=lambda i: quality(selected[i]))
    drop = set(indices_by_q[:excess])
    return [p for i, p in enumerate(selected) if i not in drop]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, help="Calendar year (e.g. 2024)")
    ap.add_argument("--from", dest="from_", help="ISO date YYYY-MM-DD (alternative to --year)")
    ap.add_argument("--to", help="ISO date YYYY-MM-DD")
    ap.add_argument("--count", type=int, default=100, help="Target photo count (default 100)")
    ap.add_argument("--album", help="Album name (default: 'Yearbook YYYY')")
    ap.add_argument("--holidays", default="se",
                    help="Country codes for holiday detection, comma-separated. "
                         "Examples: se, us, se,us. Use 'none' to disable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show discovery + selection plan without creating an album")
    args = ap.parse_args()

    start, end = parse_date_range(args)
    print(f"Range: {start.date()} – {end.date()}")

    print("Reading Photos library...", flush=True)
    db = osxphotos.PhotosDB()
    photos = db.photos(from_date=start, to_date=end)
    if not photos:
        print("No photos in range."); return

    # Holidays
    if args.holidays.lower() == "none":
        holiday_map = {}
    else:
        codes = [c.strip() for c in args.holidays.split(",") if c.strip()]
        years = sorted({start.year, end.year})
        holiday_map = expand_holidays(codes, years)

    # Bucket + plan
    buckets = bucket_photos(photos, holiday_map)
    print_discovery(photos, buckets)
    targets = allocate(buckets, args.count)
    print_plan(targets, buckets)

    # Select
    selected = pick_from_buckets(buckets, targets)
    selected = trim_to_budget(selected, args.count)
    print(f"\nFinal selection: {len(selected)} photos")

    if args.dry_run:
        print("\n(dry-run — not creating album. Top picks:)")
        for p in sorted(selected, key=lambda p: p.date)[:10]:
            label = p.place.name if p.place and p.place.name else "—"
            print(f"  {p.date.strftime('%Y-%m-%d %H:%M')}  q={quality(p):+.2f}  {label[:60]}")
        if len(selected) > 10:
            print(f"  ... and {len(selected) - 10} more")
        return

    album_name = args.album or f"Yearbook {start.year}" if start.year == end.year else f"Yearbook {start.date()}–{end.date()}"
    print(f"\nCreating album '{album_name}' in Photos…", flush=True)
    n = create_album_with(album_name, [p.uuid for p in selected])
    print(f"Done. Added {n} photos.")


if __name__ == "__main__":
    main()
