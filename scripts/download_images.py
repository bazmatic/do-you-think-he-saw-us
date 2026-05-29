#!/usr/bin/env python3
"""Fetch random JPEGs from Lorem Picsum into ./images/.

Idempotent: existing files are skipped. Filenames follow image-NNN.jpg
to stay consistent with the existing collection.

Usage:
    .venv/bin/python scripts/download_images.py            # 1000 images, 800x600
    .venv/bin/python scripts/download_images.py --count 50
    .venv/bin/python scripts/download_images.py --count 50 --size 512x512
    .venv/bin/python scripts/download_images.py --dir labels/sample
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"size must be WIDTHxHEIGHT, got {s!r}") from exc


def fetch_one(out_path: Path, width: int, height: int) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return f"skip {out_path.name}"
    url = f"https://picsum.photos/{width}/{height}"
    req = urllib.request.Request(url, headers={"User-Agent": "dino-play/download_images"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out_path)
    return f"got  {out_path.name} ({len(data) // 1024} KB)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--count", type=int, default=1000, help="number of images (default: 1000)")
    p.add_argument("--size", type=parse_size, default=(800, 600), help="WIDTHxHEIGHT (default: 800x600)")
    p.add_argument("--dir", type=Path, default=Path("images"), help="output directory (default: images)")
    p.add_argument("--workers", type=int, default=8, help="concurrent downloads (default: 8)")
    args = p.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)
    width, height = args.size
    digits = max(3, len(str(args.count)))
    paths = [args.dir / f"image-{i:0{digits}d}.jpg" for i in range(1, args.count + 1)]

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, p, width, height): p for p in paths}
        try:
            for fut in as_completed(futures):
                done += 1
                msg = fut.result()
                print(f"[{done}/{args.count}] {msg}", flush=True)
        except KeyboardInterrupt:
            print("interrupted; partial set kept", file=sys.stderr)
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
