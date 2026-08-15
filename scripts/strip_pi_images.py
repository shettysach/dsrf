#!/usr/bin/env python3
"""Create a readable Pi session copy with image payloads removed.

The output is for inspection only: it is intentionally not resumable by Pi.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO


def main() -> None:
    args = _arguments()
    if args.output is not None and args.input != "-" and args.input == args.output:
        raise SystemExit("Refusing to overwrite the original Pi session")

    source = sys.stdin if args.input == "-" else Path(args.input).open()
    destination = (
        sys.stdout if args.output is None or args.output == "-" else Path(args.output).open("w")
    )
    try:
        stripped = _write_session(source, destination, pretty=args.pretty)
    finally:
        if source is not sys.stdin:
            source.close()
        if destination is not sys.stdout:
            destination.close()
    print(f"Removed {stripped} image payload(s).", file=sys.stderr)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Pi session JSONL file, or - for stdin")
    parser.add_argument(
        "-o",
        "--output",
        help="Destination file, or - for stdout (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent each JSON record for interactive reading",
    )
    return parser.parse_args()


def _write_session(source: TextIO, destination: TextIO, *, pretty: bool) -> int:
    stripped = 0
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        cleaned, count = _strip_images(record)
        stripped += count
        if pretty:
            json.dump(cleaned, destination, ensure_ascii=False, indent=2)
        else:
            json.dump(cleaned, destination, ensure_ascii=False, separators=(",", ":"))
        destination.write("\n")
    return stripped


def _strip_images(value: Any) -> tuple[Any, int]:
    if isinstance(value, list):
        cleaned: list[Any] = []
        count = 0
        for item in value:
            stripped_item, stripped_count = _strip_images(item)
            cleaned.append(stripped_item)
            count += stripped_count
        return cleaned, count
    if not isinstance(value, dict):
        return value, 0
    if value.get("type") == "image":
        return _image_placeholder(value), 1

    cleaned: dict[str, Any] = {}
    count = 0
    for key, item in value.items():
        stripped_item, stripped_count = _strip_images(item)
        cleaned[key] = stripped_item
        count += stripped_count
    return cleaned, count


def _image_placeholder(image: dict[str, Any]) -> dict[str, Any]:
    placeholder: dict[str, Any] = {"type": "image", "omitted": True}
    for key in ("mimeType", "fileName", "size"):
        if key in image:
            placeholder[key] = image[key]
    return placeholder


if __name__ == "__main__":
    main()
