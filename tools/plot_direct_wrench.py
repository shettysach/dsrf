#!/usr/bin/env python3
"""Render DirectController root-tracking diagnostics as a PNG or JPG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("/tmp/direct-wrench.csv"),
        help="diagnostic CSV path (default: /tmp/direct-wrench.csv)",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        default=Path("/tmp/direct-wrench.png"),
        help="output image path; use a .png or .jpg extension",
    )
    args = parser.parse_args()

    if args.output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        parser.error("output_path must end in .png, .jpg, or .jpeg")
    rows = _read_rows(args.csv_path)
    if not rows:
        raise ValueError(f"No samples found in {args.csv_path}")

    time_s = _column(rows, "time_s")
    discontinuous_time_s = _discontinuous(rows, "time_s")
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 11), layout="constrained")
    _plot_components(axes[0], discontinuous_time_s, rows, "force", "Force (N)")
    axes[0].plot(discontinuous_time_s, _discontinuous(rows, "force_norm"), "k--", label="norm")
    _plot_clipping(axes[0], time_s, rows, "force_clipped")
    _plot_components(axes[1], discontinuous_time_s, rows, "torque", "Torque (Nm)")
    axes[1].plot(discontinuous_time_s, _discontinuous(rows, "torque_norm"), "k--", label="norm")
    _plot_clipping(axes[1], time_s, rows, "torque_clipped")
    _plot_components(axes[2], discontinuous_time_s, rows, "err", "Position error (m)")
    _plot_components(axes[3], discontinuous_time_s, rows, "rot_err", "Orientation error (rad)")

    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.75)
        axis.grid(alpha=0.3)
        axis.legend(ncol=4, loc="upper right")
    axes[-1].set_xlabel("Reference time (s)")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=160)
    print(f"Wrote {args.output_path}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _discontinuous(rows: list[dict[str, str]], name: str) -> list[float]:
    values: list[float] = []
    previous_observation_id: str | None = None
    for row in rows:
        if previous_observation_id is not None and row["observation_id"] != previous_observation_id:
            values.append(float("nan"))
        values.append(float(row[name]))
        previous_observation_id = row["observation_id"]
    return values


def _plot_components(
    axis: plt.Axes,
    time_s: list[float],
    rows: list[dict[str, str]],
    prefix: str,
    ylabel: str,
) -> None:
    for component in "xyz":
        axis.plot(time_s, _discontinuous(rows, f"{prefix}_{component}"), label=component)
    axis.set_ylabel(ylabel)


def _plot_clipping(
    axis: plt.Axes,
    time_s: list[float],
    rows: list[dict[str, str]],
    column: str,
) -> None:
    clipped = [
        time for time, row in zip(time_s, rows, strict=True) if row[column] == "True"
    ]
    if clipped:
        axis.scatter(clipped, [0.0] * len(clipped), color="red", marker="x", label="clipped")


if __name__ == "__main__":
    main()
