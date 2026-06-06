"""
Generate PMD experiment patterns for a 27-inch 2560 x 1440 LCD.

Assumption:
    The 27-inch display is a standard 16:9 monitor.

Output folder:
    PMD实验预备图片_2560x1440_27英寸

Generated content:
    01_屏幕外参标定图
    02_曝光与对焦检查图
    03_X方向四步相移条纹
    04_Y方向四步相移条纹
    05_X方向GrayCode
    06_Y方向GrayCode
    07_完整采集序列
    display_patterns_2560x1440_27inch.py
    参数说明.txt
    采集顺序.csv

Run:
    python generate_pmd_patterns_2560x1440_27inch.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


LCD_W = 2560
LCD_H = 1440
DIAGONAL_INCH = 27.0
ASPECT_W = 16.0
ASPECT_H = 9.0

DIAGONAL_MM = DIAGONAL_INCH * 25.4
LCD_W_MM = DIAGONAL_MM * ASPECT_W / np.sqrt(ASPECT_W**2 + ASPECT_H**2)
LCD_H_MM = DIAGONAL_MM * ASPECT_H / np.sqrt(ASPECT_W**2 + ASPECT_H**2)

# A stable first-choice period for 2560 x 1440. It gives 32 periods in X and
# 18 periods in Y, both easy for Gray Code decoding.
FRINGE_PERIOD_PX = 36

GRAY_LOW = 30
GRAY_HIGH = 225
SINE_BACKGROUND = 127.0
SINE_MODULATION = 95.0

ROOT = Path("PMD实验预备图片_2560x1440_27英寸")


DISPLAY_SCRIPT = r'''"""
Display 2560 x 1440 PMD patterns on monitor 2.

This script only displays images. It does not control the camera.

Typical command, when monitor 2 is on the right side of a 1920-wide main screen:
    python display_patterns_2560x1440_27inch.py --screen-x 1920 --screen-y 0

If monitor 2 is on the right side of a 2560-wide main screen:
    python display_patterns_2560x1440_27inch.py --screen-x 2560 --screen-y 0

Keyboard:
    Space / Right arrow : next image
    Left arrow          : previous image
    A                   : toggle auto play
    R                   : restart
    + / =               : increase interval
    -                   : decrease interval
    Esc / Q             : quit
"""

from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from pathlib import Path


def read_sequence(pattern_root: Path) -> list[Path]:
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
        screen_x: int,
        screen_y: int,
        width: int,
        height: int,
        interval_ms: int,
        manual: bool,
        show_status: bool,
    ) -> None:
        self.root = root
        self.images = images
        self.index = 0
        self.width = width
        self.height = height
        self.interval_ms = interval_ms
        self.auto = not manual
        self.show_status = show_status
        self.after_id: str | None = None
        self.current_photo: tk.PhotoImage | None = None

        root.title("PMD Pattern Display 2560x1440")
        root.geometry(f"{width}x{height}+{screen_x}+{screen_y}")
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
        if img_w != self.width or img_h != self.height:
            raise RuntimeError(
                "\n图片尺寸与窗口尺寸不一致，已停止，避免 PMD 图案缩放。\n"
                f"图片: {path.name}, 尺寸: {img_w} x {img_h}\n"
                f"窗口: {self.width} x {self.height}\n"
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
    parser = argparse.ArgumentParser(description="Display 2560x1440 PMD patterns on monitor 2.")
    parser.add_argument("--pattern-root", type=Path, default=Path("."), help="Pattern root folder.")
    parser.add_argument("--screen-x", type=int, default=1920, help="Monitor 2 left-top x coordinate.")
    parser.add_argument("--screen-y", type=int, default=0, help="Monitor 2 left-top y coordinate.")
    parser.add_argument("--width", type=int, default=2560, help="Monitor 2 native width.")
    parser.add_argument("--height", type=int, default=1440, help="Monitor 2 native height.")
    parser.add_argument("--interval-ms", type=int, default=800, help="Auto-play interval in milliseconds.")
    parser.add_argument("--manual", action="store_true", help="Start in manual mode.")
    parser.add_argument("--show-status", action="store_true", help="Show yellow status text overlay.")
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
            manual=args.manual,
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
'''


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(img)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(path.suffix, out)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(str(path))


def gray_code(x: np.ndarray) -> np.ndarray:
    return x ^ (x >> 1)


def make_sine(direction: str, period_px: int, phase: float) -> np.ndarray:
    yy, xx = np.mgrid[0:LCD_H, 0:LCD_W]
    coord = xx if direction.upper() == "X" else yy
    img = SINE_BACKGROUND + SINE_MODULATION * np.cos(2.0 * np.pi * coord / period_px + phase)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_gray_bit(direction: str, period_px: int, bit_index: int, bits: int, inverse: bool = False) -> np.ndarray:
    yy, xx = np.mgrid[0:LCD_H, 0:LCD_W]
    coord = xx if direction.upper() == "X" else yy
    total = LCD_W if direction.upper() == "X" else LCD_H
    n_periods = int(np.ceil(total / period_px))
    period_index = np.clip(np.floor(coord / period_px).astype(np.int32), 0, n_periods - 1)
    g = gray_code(period_index)
    bit = (g >> (bits - bit_index - 1)) & 1
    if inverse:
        bit = 1 - bit
    return np.where(bit > 0, GRAY_HIGH, GRAY_LOW).astype(np.uint8)


def make_checkerboard(square_mm: float = 30.0, squares_x: int = 15, squares_y: int = 9) -> tuple[np.ndarray, str]:
    pp_x = LCD_W_MM / LCD_W
    pp_y = LCD_H_MM / LCD_H
    square_px_x = int(round(square_mm / pp_x))
    square_px_y = int(round(square_mm / pp_y))
    board_w = squares_x * square_px_x
    board_h = squares_y * square_px_y
    x0 = (LCD_W - board_w) // 2
    y0 = (LCD_H - board_h) // 2

    img = np.full((LCD_H, LCD_W), 245, dtype=np.uint8)
    for iy in range(squares_y):
        for ix in range(squares_x):
            if (ix + iy) % 2 == 0:
                x1 = x0 + ix * square_px_x
                y1 = y0 + iy * square_px_y
                img[y1 : y1 + square_px_y, x1 : x1 + square_px_x] = 25
    note = (
        f"棋盘格: {squares_x} x {squares_y} 个方格, "
        f"内角点: {squares_x - 1} x {squares_y - 1}, "
        f"方格边长约 {square_mm:.3f} mm, "
        f"像素尺寸约 {square_px_x} x {square_px_y} px"
    )
    return img, note


def make_circle_grid(spacing_mm: float = 30.0, cols: int = 15, rows: int = 9) -> tuple[np.ndarray, str]:
    pp_x = LCD_W_MM / LCD_W
    pp_y = LCD_H_MM / LCD_H
    spacing_px_x = int(round(spacing_mm / pp_x))
    spacing_px_y = int(round(spacing_mm / pp_y))
    radius_px = int(round(4.0 / pp_x))

    grid_w = (cols - 1) * spacing_px_x
    grid_h = (rows - 1) * spacing_px_y
    x0 = (LCD_W - grid_w) // 2
    y0 = (LCD_H - grid_h) // 2

    img = np.full((LCD_H, LCD_W), 245, dtype=np.uint8)
    for iy in range(rows):
        for ix in range(cols):
            x = x0 + ix * spacing_px_x
            y = y0 + iy * spacing_px_y
            cv2.circle(img, (x, y), radius_px, 20, -1, lineType=cv2.LINE_AA)
    note = (
        f"对称圆点阵列: {cols} x {rows}, "
        f"圆心间距约 {spacing_mm:.3f} mm, "
        f"像素间距约 {spacing_px_x} x {spacing_px_y} px, "
        f"圆半径约 {radius_px} px"
    )
    return img, note


def make_border_points() -> np.ndarray:
    img = np.zeros((LCD_H, LCD_W), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (LCD_W - 1, LCD_H - 1), 255, 5)
    for x, y in [(0, 0), (LCD_W - 1, 0), (0, LCD_H - 1), (LCD_W - 1, LCD_H - 1), (LCD_W // 2, LCD_H // 2)]:
        cv2.drawMarker(img, (x, y), 255, markerType=cv2.MARKER_CROSS, markerSize=90, thickness=5)
        cv2.circle(img, (x, y), 36, 255, 4, lineType=cv2.LINE_AA)
    return img


def save_sequence_copy(sequence: list[tuple[str, Path]], sequence_dir: Path) -> None:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, src) in enumerate(sequence, 1):
        shutil.copy2(src, sequence_dir / f"{i:03d}_{src.name}")


def main() -> None:
    display_script_path = ROOT / "display_patterns_2560x1440_27inch.py"
    if display_script_path.exists():
        display_script_text = display_script_path.read_text(encoding="utf-8")
    else:
        display_script_text = DISPLAY_SCRIPT

    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)

    dirs = {
        "calib": ROOT / "01_屏幕外参标定图",
        "check": ROOT / "02_曝光与对焦检查图",
        "x_phase": ROOT / "03_X方向四步相移条纹",
        "y_phase": ROOT / "04_Y方向四步相移条纹",
        "x_gray": ROOT / "05_X方向GrayCode",
        "y_gray": ROOT / "06_Y方向GrayCode",
        "sequence": ROOT / "07_完整采集序列",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    sequence: list[tuple[str, Path]] = []
    notes: list[str] = []

    checker, checker_note = make_checkerboard()
    circle, circle_note = make_circle_grid()
    border = make_border_points()
    for name, img in [
        ("屏幕棋盘格_30mm_14x8内角点.png", checker),
        ("屏幕圆点阵列_30mm_15x9.png", circle),
        ("屏幕边界与中心检查图.png", border),
    ]:
        imwrite_unicode(dirs["calib"] / name, img)
    notes.extend([checker_note, circle_note])

    check_patterns = {
        "全黑_曝光检查.png": np.zeros((LCD_H, LCD_W), dtype=np.uint8),
        "全白_曝光检查.png": np.full((LCD_H, LCD_W), 255, dtype=np.uint8),
        "中灰_相机曝光建议图.png": np.full((LCD_H, LCD_W), 127, dtype=np.uint8),
        "水平灰度渐变.png": np.tile(np.linspace(0, 255, LCD_W, dtype=np.uint8), (LCD_H, 1)),
        "垂直灰度渐变.png": np.tile(np.linspace(0, 255, LCD_H, dtype=np.uint8)[:, None], (1, LCD_W)),
    }
    for name, img in check_patterns.items():
        imwrite_unicode(dirs["check"] / name, img)

    phase_steps = [
        ("第1步_0deg", 0.0),
        ("第2步_90deg", 0.5 * np.pi),
        ("第3步_180deg", np.pi),
        ("第4步_270deg", 1.5 * np.pi),
    ]
    for label, phase in phase_steps:
        path = dirs["x_phase"] / f"X方向四步相移_{label}.png"
        imwrite_unicode(path, make_sine("X", FRINGE_PERIOD_PX, phase))
        sequence.append(("X方向相移", path))
    for label, phase in phase_steps:
        path = dirs["y_phase"] / f"Y方向四步相移_{label}.png"
        imwrite_unicode(path, make_sine("Y", FRINGE_PERIOD_PX, phase))
        sequence.append(("Y方向相移", path))

    x_periods = int(np.ceil(LCD_W / FRINGE_PERIOD_PX))
    y_periods = int(np.ceil(LCD_H / FRINGE_PERIOD_PX))
    x_bits = int(np.ceil(np.log2(x_periods)))
    y_bits = int(np.ceil(np.log2(y_periods)))

    for bit in range(x_bits):
        path = dirs["x_gray"] / f"X方向GrayCode_第{bit + 1}位.png"
        inv_path = dirs["x_gray"] / f"X方向GrayCode_第{bit + 1}位_反码.png"
        imwrite_unicode(path, make_gray_bit("X", FRINGE_PERIOD_PX, bit, x_bits, False))
        imwrite_unicode(inv_path, make_gray_bit("X", FRINGE_PERIOD_PX, bit, x_bits, True))
        sequence.append(("X方向GrayCode", path))
        sequence.append(("X方向GrayCode反码", inv_path))

    for bit in range(y_bits):
        path = dirs["y_gray"] / f"Y方向GrayCode_第{bit + 1}位.png"
        inv_path = dirs["y_gray"] / f"Y方向GrayCode_第{bit + 1}位_反码.png"
        imwrite_unicode(path, make_gray_bit("Y", FRINGE_PERIOD_PX, bit, y_bits, False))
        imwrite_unicode(inv_path, make_gray_bit("Y", FRINGE_PERIOD_PX, bit, y_bits, True))
        sequence.append(("Y方向GrayCode", path))
        sequence.append(("Y方向GrayCode反码", inv_path))

    save_sequence_copy(sequence, dirs["sequence"])

    with (ROOT / "采集顺序.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "用途", "文件名"])
        for i, (purpose, path) in enumerate(sequence, 1):
            writer.writerow([i, purpose, path.name])

    pp_x = LCD_W_MM / LCD_W
    pp_y = LCD_H_MM / LCD_H
    info = [
        "PMD 实验预备图片参数说明 - 2560x1440 27英寸",
        "",
        "【LCD】",
        f"LCD 分辨率: {LCD_W} x {LCD_H} px",
        f"LCD 对角线: {DIAGONAL_INCH:.3f} inch",
        "LCD 比例: 16:9",
        f"按 27 英寸 16:9 计算的有效宽度: {LCD_W_MM:.6f} mm",
        f"按 27 英寸 16:9 计算的有效高度: {LCD_H_MM:.6f} mm",
        f"LCD 水平像素物理尺寸 pp_x = {pp_x:.9f} mm/px",
        f"LCD 垂直像素物理尺寸 pp_y = {pp_y:.9f} mm/px",
        "",
        "【条纹与编码】",
        f"条纹周期 T = {FRINGE_PERIOD_PX} px",
        f"条纹周期物理长度 X = {FRINGE_PERIOD_PX * pp_x:.6f} mm",
        f"条纹周期物理长度 Y = {FRINGE_PERIOD_PX * pp_y:.6f} mm",
        f"四步相移灰度范围: {SINE_BACKGROUND - SINE_MODULATION:.0f} 到 {SINE_BACKGROUND + SINE_MODULATION:.0f}",
        f"X方向周期数 ceil(2560/T) = {x_periods}, Gray Code 位数 = {x_bits}",
        f"Y方向周期数 ceil(1440/T) = {y_periods}, Gray Code 位数 = {y_bits}",
        f"完整采集序列图片数 = {len(sequence)}",
        "Gray Code 已生成原码和反码，建议都采集。",
        "",
        "【屏幕外参标定图】",
        *notes,
        "",
        "【显示程序】",
        "display_patterns_2560x1440_27inch.py 已放在本文件夹。",
        "若屏幕2在主屏右侧，且主屏宽度为1920:",
        "    python display_patterns_2560x1440_27inch.py --screen-x 1920 --screen-y 0",
        "若主屏宽度为2560:",
        "    python display_patterns_2560x1440_27inch.py --screen-x 2560 --screen-y 0",
        "",
        "【实验提醒】",
        "1. 必须 1:1 原始像素显示，不要缩放。",
        "2. Windows 显示缩放建议设为 100%。",
        "3. 关闭护眼、HDR、自动亮度、动态对比度、色彩增强。",
        "4. 采集时锁定相机曝光、增益、白平衡、光圈、焦距。",
    ]
    (ROOT / "参数说明.txt").write_text("\n".join(info), encoding="utf-8")
    (ROOT / "display_patterns_2560x1440_27inch.py").write_text(display_script_text, encoding="utf-8")
    print(f"已生成: {ROOT.resolve()}")
    print(f"LCD size: {LCD_W_MM:.3f} mm x {LCD_H_MM:.3f} mm")
    print(f"pp_x={pp_x:.9f} mm/px, pp_y={pp_y:.9f} mm/px")
    print(f"sequence images: {len(sequence)}")


if __name__ == "__main__":
    main()
