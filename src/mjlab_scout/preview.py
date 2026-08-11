from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def main() -> None:
    parser = argparse.ArgumentParser(description="Show an MJLab Scout view")
    parser.add_argument("image", type=Path)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--view", default="overview")
    args = parser.parse_args()

    try:
        with Image.open(args.image) as source:
            image = source.copy()

        window = tk.Tk()
        window.title(f"MJLab Scout — VLM view: {args.view}")
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(window, image=photo, borderwidth=0)
        label.image = photo  # ty: ignore[unresolved-attribute]
        label.pack()
        window.after(max(1, round(args.duration * 1000)), window.destroy)
        window.mainloop()
    finally:
        args.image.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
