"""
Generate LCD images needed for a monocular PMD experiment.

Output images are generated at the LCD native resolution:
    1920 x 1080

Generated image groups:
    01_屏幕外参标定图
    02_曝光与对焦检查图
    03_X方向四步相移条纹
    04_Y方向四步相移条纹
    05_X方向GrayCode
    06_Y方向GrayCode
    07_完整采集序列

Run:
    python generate_pmd_experiment_patterns.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


LCD_W = 1920
LCD_H = 1080
LCD_W_MM = 344.0
LCD_H_MM = 194.0

FRINGE_PERIOD_PX = 64
GRAY_LOW = 30
GRAY_HIGH = 225
SINE_BACKGROUND = 127.0
SINE_MODULATION = 95.0

ROOT = Path("PMD实验预备图片_1920x1080")


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


def make_checkerboard(square_mm: float = 20.0, squares_x: int = 13, squares_y: int = 8) -> tuple[np.ndarray, str]:
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


def make_circle_grid(spacing_mm: float = 20.0, cols: int = 13, rows: int = 7) -> tuple[np.ndarray, str]:
    pp_x = LCD_W_MM / LCD_W
    pp_y = LCD_H_MM / LCD_H
    spacing_px_x = int(round(spacing_mm / pp_x))
    spacing_px_y = int(round(spacing_mm / pp_y))
    radius_px = int(round(2.8 / pp_x))

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
    cv2.rectangle(img, (0, 0), (LCD_W - 1, LCD_H - 1), 255, 4)
    for x, y in [(0, 0), (LCD_W - 1, 0), (0, LCD_H - 1), (LCD_W - 1, LCD_H - 1), (LCD_W // 2, LCD_H // 2)]:
        cv2.drawMarker(img, (x, y), 255, markerType=cv2.MARKER_CROSS, markerSize=70, thickness=4)
        cv2.circle(img, (x, y), 28, 255, 3, lineType=cv2.LINE_AA)
    return img


def save_sequence_copy(sequence: list[tuple[str, Path]], sequence_dir: Path) -> None:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, src) in enumerate(sequence, 1):
        dst = sequence_dir / f"{i:03d}_{src.name}"
        shutil.copy2(src, dst)


def main() -> None:
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
    calib_files = [
        ("屏幕棋盘格_20mm_12x7内角点.png", checker),
        ("屏幕圆点阵列_20mm_13x7.png", circle),
        ("屏幕边界与中心检查图.png", border),
    ]
    for name, img in calib_files:
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
        imwrite_unicode(path, make_gray_bit("X", FRINGE_PERIOD_PX, bit, x_bits, inverse=False))
        imwrite_unicode(inv_path, make_gray_bit("X", FRINGE_PERIOD_PX, bit, x_bits, inverse=True))
        sequence.append(("X方向GrayCode", path))
        sequence.append(("X方向GrayCode反码", inv_path))

    for bit in range(y_bits):
        path = dirs["y_gray"] / f"Y方向GrayCode_第{bit + 1}位.png"
        inv_path = dirs["y_gray"] / f"Y方向GrayCode_第{bit + 1}位_反码.png"
        imwrite_unicode(path, make_gray_bit("Y", FRINGE_PERIOD_PX, bit, y_bits, inverse=False))
        imwrite_unicode(inv_path, make_gray_bit("Y", FRINGE_PERIOD_PX, bit, y_bits, inverse=True))
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
    focal_px = 25.0 / 0.00345
    info = [
        "PMD 实验预备图片参数说明",
        "",
        "【相机与镜头】",
        "相机型号: MER2-502-79U3C",
        "相机分辨率: 2448 x 2048 px",
        "相机像元尺寸: 3.45 um x 3.45 um",
        "镜头型号: HN-2516-5M-C2/3X",
        "镜头焦距: 25 mm",
        f"按像元尺寸估算焦距: f ≈ {focal_px:.2f} px",
        "",
        "【LCD】",
        f"LCD 图片分辨率: {LCD_W} x {LCD_H} px",
        f"LCD 有效显示区域: {LCD_W_MM:.3f} mm x {LCD_H_MM:.3f} mm",
        f"LCD 水平像素物理尺寸 pp_x = {pp_x:.9f} mm/px",
        f"LCD 垂直像素物理尺寸 pp_y = {pp_y:.9f} mm/px",
        "",
        "【条纹】",
        f"四步相移灰度范围: {SINE_BACKGROUND - SINE_MODULATION:.0f} 到 {SINE_BACKGROUND + SINE_MODULATION:.0f}",
        f"条纹周期 T = {FRINGE_PERIOD_PX} px",
        f"X 方向周期数 ceil(1920/T) = {x_periods}, Gray Code 位数 = {x_bits}",
        f"Y 方向周期数 ceil(1080/T) = {y_periods}, Gray Code 位数 = {y_bits}",
        "Gray Code 已生成原码和反码，建议都采集，用反码做阈值鲁棒判断。",
        "",
        "【屏幕外参标定图】",
        *notes,
        "",
        "【实验提醒】",
        "1. 显示图片时请关闭缩放、护眼、HDR、自动亮度、色彩增强，确保 1:1 像素显示。",
        "2. PMD 采集时相机焦距、光圈、曝光时间、增益、白平衡必须锁定。",
        "3. 条纹图不要过曝，建议先用中灰图和四步条纹第1步检查直方图。",
        "4. 相机不必拍到整个 LCD，只要有效测量区域内能看到完整编码且能解出绝对相位即可。",
        "5. 第一次实验建议先采参考平面镜两遍，做参考-参考差分，确认重建接近零。",
    ]
    (ROOT / "参数说明.txt").write_text("\n".join(info), encoding="utf-8")
    print(f"图片已生成: {ROOT.resolve()}")
    print(f"采集序列图片数量: {len(sequence)}")
    print(f"LCD pp_x={pp_x:.9f} mm/px, pp_y={pp_y:.9f} mm/px")


if __name__ == "__main__":
    main()
