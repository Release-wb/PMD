"""
Display 2560 x 1440 PMD patterns on monitor 2.

Default behavior:
    1. Automatically chooses monitor 2: the first non-primary monitor.
    2. Displays images 1:1 fullscreen on that monitor.
    3. Auto-plays the sequence once.
    4. Stops on the last image. It does not loop.

Run:
    python display_patterns_2560x1440_27inch.py

Keyboard:
    Space / Right arrow : next image
    Left arrow          : previous image
    A                   : toggle auto play
    R                   : restart from first image
    + / =               : increase auto interval
    -                   : decrease auto interval
    Esc / Q             : quit
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


@dataclass
class MonitorInfo:
    left: int
    top: int
    right: int
    bottom: int
    is_primary: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def list_monitors() -> list[MonitorInfo]:
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        ]

    user32 = ctypes.windll.user32
    monitors: list[MonitorInfo] = []
    MONITORINFOF_PRIMARY = 1

    def callback(hmonitor, _hdc, _rect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            return True
        rect = info.rcMonitor
        monitors.append(
            MonitorInfo(
                left=int(rect.left),
                top=int(rect.top),
                right=int(rect.right),
                bottom=int(rect.bottom),
                is_primary=bool(info.dwFlags & MONITORINFOF_PRIMARY),
            )
        )
        return True

    enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_double,
    )
    user32.EnumDisplayMonitors(0, 0, enum_proc(callback), 0)
    return monitors


def choose_monitor(monitor_number: int) -> MonitorInfo:
    monitors = list_monitors()
    if not monitors:
        raise RuntimeError("没有检测到显示器。")

    primary = next((m for m in monitors if m.is_primary), monitors[0])
    non_primary = [m for m in monitors if not m.is_primary]
    non_primary.sort(key=lambda m: (m.left, m.top))

    if monitor_number == 1:
        return primary
    if non_primary:
        idx = min(max(monitor_number - 2, 0), len(non_primary) - 1)
        return non_primary[idx]
    return primary


def read_sequence(pattern_root: Path) -> list[Path]:
    seq_dir = pattern_root / "07_完整采集序列"
    images = sorted(seq_dir.glob("*.png")) if seq_dir.exists() else []
    if images:
        return images

    csv_path = pattern_root / "采集顺序.csv"
    if csv_path.exists():
        images = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["序号"])
                filename = row["文件名"]
                path = seq_dir / f"{idx:03d}_{filename}"
                if not path.exists():
                    raise FileNotFoundError(f"Missing pattern image: {path}")
                images.append(path)
        if images:
            return images

    raise RuntimeError(f"No pattern images found under: {pattern_root}")


class PatternPlayer:
    def __init__(
        self,
        root: tk.Tk,
        images: list[Path],
        monitor: MonitorInfo,
        interval_ms: int,
        manual: bool,
        show_status: bool,
        close_at_end: bool,
    ) -> None:
        self.root = root
        self.images = images
        self.monitor = monitor
        self.index = 0
        self.interval_ms = interval_ms
        self.auto = not manual
        self.show_status = show_status
        self.close_at_end = close_at_end
        self.after_id: str | None = None
        self.current_photo: tk.PhotoImage | None = None

        root.title("PMD Pattern Display 2560x1440")
        root.geometry(f"{monitor.width}x{monitor.height}+{monitor.left}+{monitor.top}")
        root.configure(background="black", cursor="none")
        root.overrideredirect(True)
        root.attributes("-topmost", True)

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
        root.bind("+", lambda _e: self.change_interval(100))
        root.bind("=", lambda _e: self.change_interval(100))
        root.bind("-", lambda _e: self.change_interval(-100))
        root.bind("<Escape>", lambda _e: self.quit())
        root.bind("q", lambda _e: self.quit())
        root.bind("Q", lambda _e: self.quit())

        self.show_current()
        if self.auto:
            self.schedule_next()

    def show_current(self) -> None:
        path = self.images[self.index]
        self.current_photo = tk.PhotoImage(file=str(path))
        img_w = self.current_photo.width()
        img_h = self.current_photo.height()

        if img_w != self.monitor.width or img_h != self.monitor.height:
            raise RuntimeError(
                "\n图片尺寸与屏幕2尺寸不一致，已停止，避免 PMD 图案被缩放。\n"
                f"图片: {path.name}\n"
                f"图片尺寸: {img_w} x {img_h}\n"
                f"屏幕尺寸: {self.monitor.width} x {self.monitor.height}\n"
                "请确认屏幕2分辨率是 2560 x 1440，并尽量将 Windows 缩放设为 100%。\n"
            )

        self.label.configure(image=self.current_photo)
        if self.show_status:
            mode = "AUTO" if self.auto else "MANUAL"
            self.status.configure(
                text=(
                    f"{self.index + 1:03d}/{len(self.images):03d}  {mode}  "
                    f"{self.interval_ms} ms  screen={self.monitor.width}x{self.monitor.height}"
                    f"+{self.monitor.left}+{self.monitor.top}  {path.name}"
                )
            )

    def next_image(self) -> None:
        if self.index >= len(self.images) - 1:
            self.auto = False
            if self.close_at_end:
                self.quit()
            else:
                self.show_current()
            return
        self.index += 1
        self.show_current()
        if self.auto:
            self.schedule_next()

    def prev_image(self) -> None:
        self.index = max(0, self.index - 1)
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
    parser = argparse.ArgumentParser(description="Display 2560x1440 PMD patterns on monitor 2.")
    parser.add_argument("--pattern-root", type=Path, default=Path("."), help="Pattern root folder.")
    parser.add_argument("--monitor", type=int, default=2, help="1=primary, 2=first non-primary monitor.")
    parser.add_argument("--interval-ms", type=int, default=800, help="Auto-play interval in milliseconds.")
    parser.add_argument("--manual", action="store_true", help="Start in manual mode.")
    parser.add_argument("--close-at-end", action="store_true", help="Close the window after the last image.")
    parser.add_argument("--show-status", action="store_true", help="Show yellow status text overlay.")
    return parser.parse_args()


def main() -> None:
    enable_dpi_awareness()
    args = parse_args()
    images = read_sequence(args.pattern_root)
    monitor = choose_monitor(args.monitor)
    print(
        f"使用显示器 monitor={args.monitor}: "
        f"位置=({monitor.left},{monitor.top}), "
        f"尺寸={monitor.width}x{monitor.height}, "
        f"primary={monitor.is_primary}"
    )

    root = tk.Tk()
    try:
        PatternPlayer(
            root=root,
            images=images,
            monitor=monitor,
            interval_ms=args.interval_ms,
            manual=args.manual,
            close_at_end=args.close_at_end,
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
