"""
Display PMD pattern images on monitor 2 in sequence.

This script only controls image display. It does NOT control the camera.

Recommended use on Windows:
    python display_patterns_screen2.py --screen-x 1920 --screen-y 0

If monitor 2 is placed to the right of monitor 1, --screen-x is usually the
width of monitor 1. For example:
    monitor 1: 1920 x 1080
    monitor 2: 1920 x 1080, placed on the right
    => --screen-x 1920 --screen-y 0

Keyboard:
    Space / Right arrow : next image
    Left arrow          : previous image
    A                   : toggle auto play
    R                   : restart from first image
    + / =               : increase auto interval
    -                   : decrease auto interval
    Esc / Q             : quit

Important:
    For PMD, display images 1:1 at the LCD native resolution.
    Do not stretch 1920x1080 patterns to a 2560x1440 window.
    If your monitor 2 is 2560x1440, regenerate patterns at 2560x1440.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from pathlib import Path


def read_sequence(pattern_root: Path) -> list[Path]:
    """
    Read display sequence.

    Priority:
    1. pattern_root/07_完整采集序列/*.png
    2. pattern_root/采集顺序.csv + pattern_root/07_完整采集序列
    """
    seq_dir = pattern_root / "07_完整采集序列"
    if seq_dir.exists():
        images = sorted(seq_dir.glob("*.png"))
        if images:
            return images

    csv_path = pattern_root / "采集顺序.csv"
    if csv_path.exists():
        images: list[Path] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["序号"])
                filename = row["文件名"]
                numbered = seq_dir / f"{idx:03d}_{filename}"
                plain = seq_dir / filename
                if numbered.exists():
                    images.append(numbered)
                elif plain.exists():
                    images.append(plain)
                else:
                    raise FileNotFoundError(f"Missing pattern image: {filename}")
        if images:
            return images

    raise RuntimeError(f"No pattern images found under: {pattern_root}")


class PatternPlayer:
    def __init__(
        self,
        root: tk.Tk,
        images: list[Path],
        screen_x: int,
        screen_y: int,
        width: int,
        height: int,
        interval_ms: int,
        start_auto: bool,
        show_status: bool,
    ) -> None:
        self.root = root
        self.images = images
        self.index = 0
        self.interval_ms = interval_ms
        self.auto = start_auto
        self.show_status = show_status
        self.after_id: str | None = None
        self.current_photo: tk.PhotoImage | None = None
        self.width = width
        self.height = height

        root.title("PMD Pattern Display")
        root.geometry(f"{width}x{height}+{screen_x}+{screen_y}")
        root.configure(background="black")

        # Borderless fullscreen-like window on the selected monitor.
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(cursor="none")

        self.label = tk.Label(root, bg="black", bd=0, highlightthickness=0)
        self.label.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(root, fg="yellow", bg="black", anchor="w", font=("Consolas", 14))
        if show_status:
            self.status.place(x=8, y=8)

        root.bind("<space>", lambda _e: self.next_image())
        root.bind("<Right>", lambda _e: self.next_image())
        root.bind("<Left>", lambda _e: self.prev_image())
        root.bind("a", lambda _e: self.toggle_auto())
        root.bind("A", lambda _e: self.toggle_auto())
        root.bind("r", lambda _e: self.restart())
        root.bind("R", lambda _e: self.restart())
        root.bind("<Escape>", lambda _e: self.quit())
        root.bind("q", lambda _e: self.quit())
        root.bind("Q", lambda _e: self.quit())
        root.bind("+", lambda _e: self.change_interval(100))
        root.bind("=", lambda _e: self.change_interval(100))
        root.bind("-", lambda _e: self.change_interval(-100))

        self.show_current()
        if self.auto:
            self.schedule_next()

    def show_current(self) -> None:
        path = self.images[self.index]
        # Tk 8.6 supports PNG directly through PhotoImage.
        self.current_photo = tk.PhotoImage(file=str(path))
        img_w = self.current_photo.width()
        img_h = self.current_photo.height()
        if img_w != self.width or img_h != self.height:
            raise RuntimeError(
                "\n图片尺寸和显示窗口尺寸不一致，程序已停止，避免 PMD 图案被缩放。\n"
                f"图片: {path.name}\n"
                f"图片尺寸: {img_w} x {img_h}\n"
                f"窗口尺寸: {self.width} x {self.height}\n\n"
                "解决办法:\n"
                f"1. 如果屏幕2原生分辨率是 {img_w} x {img_h}，请用 "
                f"--width {img_w} --height {img_h}\n"
                "2. 如果屏幕2原生分辨率是当前窗口尺寸，请重新生成同分辨率图案。\n"
            )
        self.label.configure(image=self.current_photo)
        if self.show_status:
            mode = "AUTO" if self.auto else "MANUAL"
            self.status.configure(
                text=f"{self.index + 1:03d}/{len(self.images):03d}  {mode}  {self.interval_ms} ms  {path.name}"
            )

    def next_image(self) -> None:
        self.index = (self.index + 1) % len(self.images)
        self.show_current()
        if self.auto:
            self.schedule_next()

    def prev_image(self) -> None:
        self.index = (self.index - 1) % len(self.images)
        self.show_current()

    def restart(self) -> None:
        self.index = 0
        self.show_current()
        if self.auto:
            self.schedule_next()

    def toggle_auto(self) -> None:
        self.auto = not self.auto
        if self.auto:
            self.schedule_next()
        elif self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.show_current()

    def change_interval(self, delta_ms: int) -> None:
        self.interval_ms = max(100, self.interval_ms + delta_ms)
        self.show_current()
        if self.auto:
            self.schedule_next()

    def schedule_next(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(self.interval_ms, self.next_image)

    def quit(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display PMD patterns on monitor 2.")
    parser.add_argument(
        "--pattern-root",
        type=Path,
        default=Path("PMD实验预备图片_1920x1080"),
        help="Pattern root folder.",
    )
    parser.add_argument("--screen-x", type=int, default=1920, help="Monitor 2 left-top x coordinate.")
    parser.add_argument("--screen-y", type=int, default=0, help="Monitor 2 left-top y coordinate.")
    parser.add_argument("--width", type=int, default=2560, help="Monitor 2 native width; must equal pattern width.")
    parser.add_argument("--height", type=int, default=1440, help="Monitor 2 native height; must equal pattern height.")
    parser.add_argument("--interval-ms", type=int, default=800, help="Auto-play interval in milliseconds.")
    parser.add_argument("--manual", action="store_true", help="Start in manual mode instead of auto play.")
    parser.add_argument("--show-status", action="store_true", help="Show yellow status text overlay. Do not use for formal PMD capture.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = read_sequence(args.pattern_root)
    root = tk.Tk()
    try:
        PatternPlayer(
            root=root,
            images=images,
            screen_x=args.screen_x,
            screen_y=args.screen_y,
            width=args.width,
            height=args.height,
            interval_ms=args.interval_ms,
            start_auto=not args.manual,
            show_status=args.show_status,
        )
        root.mainloop()
    except Exception as exc:
        try:
            root.destroy()
        except Exception:
            pass
        print(exc, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
