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
import imagehash
from PIL import Image
from pathlib import Path as _Path


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
                 min_dist_km: float = 50.0,
                 min_trip_size: int = 8,
                 min_named_persons: int = 2) -> list[Cluster]:
    if not home:
        return []
    away = [p for p in photos
            if p.location
            and p.location[0] is not None and p.location[1] is not None
            and haversine_km(p.location, home) >= min_dist_km]
    trip_clusters = []
    for group in split_by_time_gap(away, gap_hours=24):
        if len(group) < min_trip_size:
            continue
        # Trip must include at least min_named_persons distinct tagged faces
        # across all its photos (filters out hospital visits, work conferences,
        # solo errands, etc. when we want a family yearbook).
        if min_named_persons > 0:
            named: set[str] = set()
            for p in group:
                for n in (p.persons or []):
                    if n and not n.startswith("_"):
                        named.add(n)
            if len(named) < min_named_persons:
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


def is_yearbook_worthy(p, allow_videos: bool, allow_screenshots: bool,
                       allow_no_camera: bool, allow_no_gps: bool) -> bool:
    """Filter out content that doesn't belong in a printable yearbook:
    videos (don't render in books), screenshots (Pokemon/UI captures),
    photos with no camera info (downloaded images, web saves), and photos
    without geolocation (anchorless, weak yearbook material)."""
    if getattr(p, "ismovie", False) and not allow_videos:
        return False
    if getattr(p, "screenshot", False) and not allow_screenshots:
        return False
    if not allow_no_camera:
        cam = p.exif_info.camera_make if p.exif_info else None
        if not cam:
            return False
    if not allow_no_gps:
        loc = p.location
        if not loc or loc[0] is None or loc[1] is None:
            return False
    return True


def bucket_photos(photos: list, holiday_map: dict[date, str],
                  min_trip_size: int = 8,
                  min_trip_persons: int = 2) -> Buckets:
    home = home_centroid(photos)
    b = Buckets()
    b.trips = detect_trips(photos, home, min_trip_size=min_trip_size,
                           min_named_persons=min_trip_persons)
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


_phash_cache: dict[str, imagehash.ImageHash | None] = {}
_similarity_threshold: int = 14  # mutated by main() from CLI flag


def phash_of(photo) -> imagehash.ImageHash | None:
    """Compute (and cache) a perceptual hash from the smallest local derivative.

    A small JPEG is enough — pHash internally downsamples to 8x8.
    """
    if photo.uuid in _phash_cache:
        return _phash_cache[photo.uuid]
    derivs = photo.path_derivatives or []
    # Smallest derivative is usually last; use it for speed.
    for d in reversed(derivs):
        if d and _Path(d).exists():
            try:
                with Image.open(d) as img:
                    h = imagehash.phash(img)
                _phash_cache[photo.uuid] = h
                return h
            except Exception:
                continue
    _phash_cache[photo.uuid] = None
    return None


def select_top(photos: list, n: int,
               similarity_threshold: int | None = None) -> list:
    """Pick up to `n` photos by quality, skipping near-duplicates.

    Two photos count as duplicates when their pHash hamming distance is below
    `similarity_threshold` (out of 64). 12 catches passport-style burst shots
    of the same scene; lower is more aggressive, higher is more permissive.
    Photos without a usable derivative fall through without dedup.
    """
    if n <= 0 or not photos:
        return []
    threshold = similarity_threshold if similarity_threshold is not None else _similarity_threshold
    sorted_by_q = sorted(photos, key=quality, reverse=True)
    picked: list = []
    picked_hashes: list = []
    for p in sorted_by_q:
        h = phash_of(p)
        if h is not None and any(
            ph is not None and (h - ph) < threshold
            for ph in picked_hashes
        ):
            continue
        picked.append(p)
        picked_hashes.append(h)
        if len(picked) >= n:
            break
    return picked


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
    # osxphotos exposes bare UUIDs ("BB76C4A3-…"), but Photos AppleScript ids
    # include a per-asset suffix ("/L0/001", "/V0/…" for videos). `starts with`
    # matches across both forms.
    script = f'''
    tell application "Photos"
        set newAlbum to make new album named "{safe_name}"
        set added to 0
        repeat with targetId in {uuid_list}
            set found to {{}}
            try
                set found to (every media item whose id starts with targetId)
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
            named = trip_named_persons(c)
            print(f"  {c.start.date()}–{c.end.date()}  {c.label} "
                  f"({len(c.photos)} photos, {len(named)} faces)")

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

def trip_named_persons(cluster) -> set[str]:
    """Distinct tagged-person names across all photos of a trip."""
    out: set[str] = set()
    for p in cluster.photos:
        for n in (p.persons or []):
            if n and not n.startswith("_"):
                out.add(n)
    return out


def trip_significance(cluster) -> float:
    """Weight a trip by length × tagged-people-count (capped).

    Used to apportion the trip budget: a 40-photo family trip with 5
    tagged faces ranks much higher than a 20-photo solo trip, but not
    so high that 10-person trips swallow the whole budget. Persons cap
    at 6 to keep the scale bounded; size is taken raw.
    """
    size = len(cluster.photos)
    persons = max(1, min(len(trip_named_persons(cluster)), 6))
    return size * persons


def select_with_day_spread(photos: list, n: int) -> list:
    """Distribute `n` picks proportionally across the distinct dates, then
    take top-quality (with pHash dedup) within each date.

    Without this a 5-day trip with one photogenic afternoon collapses to
    that single afternoon, missing the rest of the trip entirely.
    """
    if n <= 0 or not photos:
        return []
    by_day: dict = defaultdict(list)
    for p in photos:
        d = p.date.date() if p.date else None
        by_day[d].append(p)
    days = sorted(by_day.keys(), key=lambda d: (d is None, d))
    sizes = [len(by_day[d]) for d in days]
    total = sum(sizes)
    out: list = []
    remaining = n
    for d, sz in zip(days, sizes):
        if remaining <= 0:
            break
        share = max(1, round(n * sz / total))
        share = min(share, remaining)
        out.extend(select_top(by_day[d], share))
        remaining -= share
    return out


def pick_from_buckets(buckets: Buckets, targets: dict[str, int],
                      max_per_cluster: int, max_per_trip: int) -> list:
    selected = []

    # trips: weight each trip by size × tagged-people, distribute the
    # trip budget proportionally, then pick across the trip's days
    trips_target = targets["trips"]
    if buckets.trips and trips_target:
        weights = [trip_significance(c) for c in buckets.trips]
        total_w = sum(weights) or 1
        for c, w in zip(buckets.trips, weights):
            n = max(1, round(trips_target * w / total_w))
            n = min(n, max_per_trip)
            selected.extend(select_with_day_spread(c.photos, n))

    # holidays: distribute proportionally across distinct holidays, capped
    holiday_target = targets["holidays"]
    if buckets.holidays and holiday_target:
        by_base = defaultdict(list)
        for label, lst in buckets.holidays.items():
            base = label.replace(" (eve)", "")
            by_base[base].extend(lst)
        sizes = {k: len(v) for k, v in by_base.items()}
        total_h = sum(sizes.values())
        for k, lst in by_base.items():
            n = max(1, round(holiday_target * sizes[k] / total_h))
            n = min(n, max_per_cluster)
            selected.extend(select_top(lst, n))

    # events: take from biggest clusters first, capped per cluster
    event_target = targets["events"]
    if buckets.events and event_target:
        sorted_events = sorted(buckets.events, key=lambda c: -len(c.photos))
        per_event = max(1, event_target // max(1, min(len(sorted_events), 6)))
        remaining = event_target
        for c in sorted_events:
            if remaining <= 0:
                break
            n = min(per_event, remaining, max_per_cluster)
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


def scene_key(p) -> tuple:
    """Coarse (date, ~1km) bucket for grouping photos into 'scenes'.
    Photos lacking GPS are placed in singleton buckets (per-uuid)."""
    date_key = p.date.date() if p.date else None
    loc = p.location
    if loc and loc[0] is not None and loc[1] is not None:
        return (date_key, round(loc[0], 2), round(loc[1], 2))
    return ("singleton", p.uuid)


def scene_dedup(selected: list, keep_per_scene: int) -> list:
    """Cap how many photos can come from the same (date, ~1km area).

    A 4-hour passport-photo session at home is a single 'scene' even though
    its photos vary slightly in pose/expression — pHash is too coarse to
    catch this kind of repetition. Date+location bucketing reliably groups
    them and lets us keep just the top N by quality.

    Photos without GPS are placed in singleton groups (not deduped) so a
    library of indoor shots without geotags isn't collapsed.
    """
    if keep_per_scene <= 0:
        return selected
    groups: dict = defaultdict(list)
    for p in selected:
        groups[scene_key(p)].append(p)
    out: list = []
    for g in groups.values():
        out.extend(sorted(g, key=quality, reverse=True)[:keep_per_scene])
    return out


def global_dedup(selected: list, threshold: int) -> list:
    """Drop near-duplicates from the whole selection.

    pick_from_buckets only deduplicates within each bucket, so a portrait
    series can survive by landing partially in 'events' and partially in
    'everyday'. This pass walks the merged result in quality order and keeps
    only photos that aren't too close to anything already kept.
    """
    out: list = []
    out_hashes: list = []
    for p in sorted(selected, key=quality, reverse=True):
        h = phash_of(p)
        if h is not None and any(
            ph is not None and (h - ph) < threshold
            for ph in out_hashes
        ):
            continue
        out.append(p)
        out_hashes.append(h)
    return out


def topup_to_budget(selected: list, all_in_range: list, target: int,
                    keep_per_scene: int) -> list:
    """Fill the selection up to `target` from the highest-quality remaining
    photos, while respecting both pHash similarity and per-scene caps."""
    if len(selected) >= target:
        return selected
    have = {p.uuid for p in selected}
    out = list(selected)
    out_hashes = [phash_of(p) for p in out]
    scene_counts: dict = defaultdict(int)
    for p in out:
        scene_counts[scene_key(p)] += 1
    extras = sorted(
        (p for p in all_in_range if p.uuid not in have),
        key=quality, reverse=True,
    )
    for cand in extras:
        if len(out) >= target:
            break
        if keep_per_scene > 0 and scene_counts[scene_key(cand)] >= keep_per_scene:
            continue
        h = phash_of(cand)
        if h is not None and any(
            ph is not None and (h - ph) < _similarity_threshold
            for ph in out_hashes
        ):
            continue
        out.append(cand)
        out_hashes.append(h)
        scene_counts[scene_key(cand)] += 1
    return out


def named_persons(photo) -> list[str]:
    return [n for n in (photo.persons or []) if n and not n.startswith("_")]


def rebalance_persons(selected: list, all_in_range: list,
                      target: int, max_share: float) -> list:
    """Swap dominating-person photos for under-represented ones.

    For any named person appearing in more than `max_share * target` of the
    selection, drop their lowest-quality photo and swap in the highest-quality
    unselected photo that doesn't include them. Continues until no person
    crosses the threshold or no replacement candidate is left.
    """
    if max_share <= 0 or target <= 0 or not selected:
        return selected

    selected = list(selected)
    selected_uuids = {p.uuid for p in selected}
    pool = sorted(
        (p for p in all_in_range if p.uuid not in selected_uuids),
        key=quality, reverse=True,
    )

    threshold = int(target * max_share)
    swaps = 0

    while True:
        counts = Counter()
        for p in selected:
            for n in named_persons(p):
                counts[n] += 1
        over = [(name, c) for name, c in counts.items() if c > threshold]
        if not over:
            break
        over_name, _ = max(over, key=lambda x: x[1])

        drop_idx = min(
            (i for i, p in enumerate(selected) if over_name in named_persons(p)),
            key=lambda i: quality(selected[i]),
            default=None,
        )
        if drop_idx is None:
            break

        repl_idx = next(
            (i for i, p in enumerate(pool) if over_name not in named_persons(p)),
            None,
        )
        if repl_idx is None:
            break  # no replacement candidates left

        selected[drop_idx] = pool.pop(repl_idx)
        swaps += 1

    if swaps:
        print(f"Person balance: swapped {swaps} photos to reduce dominance.")
    return selected


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
    ap.add_argument("--min-trip-size", type=int, default=8,
                    help="Minimum photos away-from-home for something to "
                         "count as a 'trip'. Lower → more day-excursions "
                         "and brief outings get included. Default: 8")
    ap.add_argument("--include-videos", action="store_true",
                    help="Include movie clips (excluded by default — they "
                         "don't render in printed books).")
    ap.add_argument("--include-screenshots", action="store_true",
                    help="Include screenshots (excluded by default).")
    ap.add_argument("--include-no-camera", action="store_true",
                    help="Include photos lacking EXIF camera info: "
                         "downloads, web saves, etc. (excluded by default).")
    ap.add_argument("--include-no-gps", action="store_true",
                    help="Include photos without geolocation. By default "
                         "non-geotagged photos are excluded — they make "
                         "weaker yearbook anchors.")
    ap.add_argument("--min-trip-persons", type=int, default=2,
                    help="A 'trip' must contain at least this many distinct "
                         "named (tagged) people across all its photos. "
                         "Excludes hospital visits, solo work conferences, "
                         "and similar non-family travel. Default: 2. "
                         "Set to 0 to disable.")
    ap.add_argument("--keep-per-scene", type=int, default=2,
                    help="Cap on photos from the same (date, ~1km area). "
                         "Catches dense scenes that pHash misses (e.g. a "
                         "passport-photo session: same wall, different "
                         "expressions but visually distinct hashes). "
                         "Default: 2")
    ap.add_argument("--similarity-threshold", type=int, default=14,
                    help="Visual similarity cutoff (pHash hamming distance, "
                         "0–64). Photos within this distance of an already-"
                         "picked one are skipped. Lower = stricter dedup. "
                         "Default: 12")
    ap.add_argument("--max-per-cluster", type=int, default=6,
                    help="Cap on how many photos one event or holiday can "
                         "contribute. Prevents a single 200-photo wedding "
                         "from dominating the selection. Default: 6")
    ap.add_argument("--max-per-trip", type=int, default=20,
                    help="Upper bound on how many photos one trip can "
                         "contribute. The actual share is computed from "
                         "trip size × tagged-people count, distributed "
                         "across all detected trips. This flag is the "
                         "ceiling; a small or low-people trip won't reach "
                         "it. Default: 20")
    ap.add_argument("--person-balance", type=float, default=0.40,
                    metavar="SHARE",
                    help="Maximum share of the final selection any single "
                         "named person may appear in (0.0–1.0). Set to 0 to "
                         "disable. Default: 0.40")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show discovery + selection plan without creating an album")
    args = ap.parse_args()

    global _similarity_threshold
    _similarity_threshold = args.similarity_threshold

    start, end = parse_date_range(args)
    print(f"Range: {start.date()} – {end.date()}")

    print("Reading Photos library...", flush=True)
    db = osxphotos.PhotosDB()
    raw_photos = db.photos(from_date=start, to_date=end)
    if not raw_photos:
        print("No photos in range."); return

    photos = [
        p for p in raw_photos
        if is_yearbook_worthy(p, args.include_videos,
                              args.include_screenshots, args.include_no_camera,
                              args.include_no_gps)
    ]
    dropped = len(raw_photos) - len(photos)
    if dropped:
        print(f"  filtered out {dropped} videos / screenshots / no-camera / no-gps "
              f"({len(photos)} remain)")

    # Holidays
    if args.holidays.lower() == "none":
        holiday_map = {}
    else:
        codes = [c.strip() for c in args.holidays.split(",") if c.strip()]
        years = sorted({start.year, end.year})
        holiday_map = expand_holidays(codes, years)

    # Bucket + plan
    buckets = bucket_photos(photos, holiday_map,
                            min_trip_size=args.min_trip_size,
                            min_trip_persons=args.min_trip_persons)
    print_discovery(photos, buckets)
    targets = allocate(buckets, args.count)
    print_plan(targets, buckets)

    # Select
    selected = pick_from_buckets(buckets, targets,
                                 args.max_per_cluster, args.max_per_trip)
    before = len(selected)
    selected = scene_dedup(selected, args.keep_per_scene)
    if len(selected) < before:
        print(f"\nScene dedup: trimmed {before - len(selected)} photos "
              f"from same-day same-place groups.")
    before = len(selected)
    selected = global_dedup(selected, _similarity_threshold)
    if len(selected) < before:
        print(f"pHash dedup: removed {before - len(selected)} visually similar.")
    selected = topup_to_budget(selected, photos, args.count, args.keep_per_scene)
    selected = trim_to_budget(selected, args.count)
    selected = rebalance_persons(selected, photos, args.count, args.person_balance)
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
