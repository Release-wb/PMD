"""
按真实实验流程组织的单目 PMD 验证仿真。

本程序对应用户给出的最小可行实验流程：
1. 相机内参标定
2. 参考平面镜标定
3. LCD 屏幕外参标定
4. 平面镜参考相位采集：四步相移 + Gray Code
5. 被测高反表面相位采集
6. 由相位差和平均参考距离 RA 估计斜率
7. DCT-Poisson 积分恢复相对面型

注意：
    第 6 步使用的是简化关系
        slope = DeltaPhi * T * pp / (4*pi*mean(RA))
    它适用于小斜率、小高度变化、参考镜与被测镜位置近似一致的验证实验。
    若要做高精度定量，应改用逐像素 C-P-Q 反射几何求法向。

依赖：
    numpy, opencv-python, matplotlib, scipy

运行：
    python pmd_realistic_workflow_simulation.py
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EPS = 1e-12


@dataclass
class Camera:
    width: int = 480
    height: int = 360
    fx: float = 550.0
    fy: float = 550.0
    cx: float = 239.5
    cy: float = 179.5

    def k(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def rays(self) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        x = (xx - self.cx) / self.fx
        y = (yy - self.cy) / self.fy
        return normalize(np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float64))


@dataclass
class Screen:
    """LCD 屏幕平面，简化为相机坐标系中的 z = z_screen 平面。"""

    z: float = 0.20
    width_m: float = 1.80
    height_m: float = 1.25
    width_px: int = 1920
    height_px: int = 1080

    @property
    def pp_x(self) -> float:
        return self.width_m / self.width_px

    @property
    def pp_y(self) -> float:
        return self.height_m / self.height_px

    def world_to_uv(self, p: np.ndarray) -> np.ndarray:
        u = (p[..., 0] + 0.5 * self.width_m) / self.pp_x
        v = (p[..., 1] + 0.5 * self.height_m) / self.pp_y
        return np.stack([u, v], axis=-1)

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        x = uv[..., 0] * self.pp_x - 0.5 * self.width_m
        y = uv[..., 1] * self.pp_y - 0.5 * self.height_m
        z = np.full_like(x, self.z, dtype=np.float64)
        return np.stack([x, y, z], axis=-1)

    def intersect(self, points: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = (self.z - points[..., 2]) / (dirs[..., 2] + EPS)
        hit = points + t[..., None] * dirs
        uv = self.world_to_uv(hit)
        valid = (
            (t > 0)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] < self.width_px)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < self.height_px)
        )
        return hit, valid


class Surface:
    """高反表面基类：z = z0 + h(x,y)。"""

    def __init__(self, z0: float = 1.0, name: str = "surface") -> None:
        self.z0 = z0
        self.name = name

    def height_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros_like(x, dtype=np.float64)

    def grad_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(x, dtype=np.float64), np.zeros_like(y, dtype=np.float64)

    def intersect(self, rays: np.ndarray, iters: int = 9) -> np.ndarray:
        """Newton 迭代求相机射线与 z=z0+h(x,y) 的交点。"""
        dz = rays[..., 2]
        t = self.z0 / (dz + EPS)
        for _ in range(iters):
            x = t * rays[..., 0]
            y = t * rays[..., 1]
            h = self.height_xy(x, y)
            hx, hy = self.grad_xy(x, y)
            f = t * dz - self.z0 - h
            df = dz - hx * rays[..., 0] - hy * rays[..., 1]
            t -= f / (df + EPS)
        return t[..., None] * rays

    def normal(self, p: np.ndarray) -> np.ndarray:
        hx, hy = self.grad_xy(p[..., 0], p[..., 1])
        return normalize(np.stack([-hx, -hy, np.ones_like(hx)], axis=-1))


class PlaneSurface(Surface):
    pass


class GaussianDefectSurface(Surface):
    def __init__(self, z0: float, amp: float, sigma: float, x0: float, y0: float, name: str) -> None:
        super().__init__(z0, name)
        self.amp = amp
        self.sigma = sigma
        self.x0 = x0
        self.y0 = y0

    def height_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2
        return self.amp * np.exp(-r2 / (2.0 * self.sigma**2))

    def grad_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.height_xy(x, y)
        hx = -(x - self.x0) / (self.sigma**2) * h
        hy = -(y - self.y0) / (self.sigma**2) * h
        return hx, hy


class SphericalCapSurface(Surface):
    """浅球冠面。孔径取大于视场，避免边缘截断造成相位展开困难。"""

    def __init__(self, z0: float, amp: float, radius: float, aperture: float, name: str) -> None:
        super().__init__(z0, name)
        self.amp = amp
        self.radius = radius
        self.aperture = aperture
        self.edge = np.sqrt(max(radius**2 - aperture**2, EPS))
        self.scale = amp / (radius - self.edge + EPS)

    def height_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r2 = x**2 + y**2
        cap = np.sqrt(np.maximum(self.radius**2 - r2, EPS))
        return self.scale * (cap - self.edge)

    def grad_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = x**2 + y**2
        cap = np.sqrt(np.maximum(self.radius**2 - r2, EPS))
        hx = self.scale * (-x / cap)
        hy = self.scale * (-y / cap)
        return hx, hy


class SmoothSemiDomeSurface(Surface):
    """局部平滑半圆帽面，用于最小 PMD 流程的小斜率验证。"""

    def __init__(self, z0: float, amp: float, aperture: float, x0: float, y0: float, name: str) -> None:
        super().__init__(z0, name)
        self.amp = amp
        self.aperture = aperture
        self.x0 = x0
        self.y0 = y0

    def height_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xx = x - self.x0
        yy = y - self.y0
        rho2 = (xx**2 + yy**2) / (self.aperture**2)
        base = np.maximum(1.0 - rho2, 0.0)
        return self.amp * base**2

    def grad_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx = x - self.x0
        yy = y - self.y0
        rho2 = (xx**2 + yy**2) / (self.aperture**2)
        base = np.maximum(1.0 - rho2, 0.0)
        hx = self.amp * 2.0 * base * (-2.0 * xx / self.aperture**2)
        hy = self.amp * 2.0 * base * (-2.0 * yy / self.aperture**2)
        return hx, hy


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """把相位差限制到 [-pi, pi]，避免 Gray Code 周期边界造成 2pi 跳变。"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    return normalize(d - 2.0 * np.sum(d * n, axis=-1, keepdims=True) * n)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(img)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(path.suffix, out)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    buf.tofile(str(path))


def make_grid_points(pattern_size: tuple[int, int], spacing: float) -> np.ndarray:
    cols, rows = pattern_size
    pts = []
    for i in range(rows):
        for j in range(cols):
            pts.append([(j - (cols - 1) / 2) * spacing, (i - (rows - 1) / 2) * spacing, 0.0])
    return np.asarray(pts, dtype=np.float32)


def blob_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 1500
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(params)


def render_circle_board(
    camera: Camera,
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    radius_px: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pts, _ = cv2.projectPoints(object_points, rvec, tvec, camera.k(), np.zeros(5))
    pts = pts.reshape(-1, 2)
    img = np.full((camera.height, camera.width), 245, dtype=np.float64)
    for p in pts:
        if 0 <= p[0] < camera.width and 0 <= p[1] < camera.height:
            cv2.circle(img, tuple(np.round(p).astype(int)), radius_px, 0, -1, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img += rng.normal(0.0, 0.8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), pts


def detect_grid(img: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    ok, centers = cv2.findCirclesGrid(
        img,
        pattern_size,
        flags=cv2.CALIB_CB_SYMMETRIC_GRID,
        blobDetector=blob_detector(),
    )
    return ok, centers


def draw_detection(path: Path, img: np.ndarray, pattern_size: tuple[int, int], centers: np.ndarray | None, ok: bool) -> None:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if centers is not None:
        cv2.drawChessboardCorners(vis, pattern_size, centers, ok)
    imwrite_unicode(path, vis)


def draw_reprojection(path: Path, img: np.ndarray, centers: np.ndarray, reproj: np.ndarray) -> None:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p in centers.reshape(-1, 2):
        cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0, 170, 0), -1, lineType=cv2.LINE_AA)
    for p in reproj.reshape(-1, 2):
        cv2.drawMarker(
            vis,
            tuple(np.round(p).astype(int)),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
    imwrite_unicode(path, vis)


def calibrate_camera_from_synthetic_images(true_camera: Camera, folder: Path) -> tuple[Camera, np.ndarray, float, int]:
    pattern_size = (9, 7)
    obj = make_grid_points(pattern_size, 0.035)
    poses = [
        ([0.00, 0.00, 0.00], [0.00, 0.00, 0.75]),
        ([0.16, -0.12, 0.08], [-0.05, -0.03, 0.80]),
        ([-0.16, 0.10, -0.10], [0.05, 0.03, 0.84]),
        ([0.12, 0.18, 0.14], [-0.04, 0.04, 0.88]),
        ([-0.20, -0.14, 0.10], [0.05, -0.04, 0.91]),
        ([0.08, -0.20, -0.16], [-0.03, 0.02, 0.72]),
        ([-0.12, 0.22, 0.04], [0.04, -0.02, 0.76]),
        ([0.22, 0.04, -0.12], [-0.06, 0.01, 0.87]),
        ([-0.08, -0.22, 0.18], [0.03, 0.05, 0.81]),
        ([0.15, 0.15, -0.18], [-0.02, -0.05, 0.83]),
        ([-0.22, 0.04, 0.16], [0.06, 0.02, 0.93]),
        ([0.05, -0.10, 0.22], [0.00, -0.04, 0.77]),
    ]
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    saved: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (rv, tv) in enumerate(poses, 1):
        img, _ = render_circle_board(true_camera, obj, np.asarray(rv, np.float64), np.asarray(tv, np.float64), 6, i)
        imwrite_unicode(folder / f"相机内参标定_圆点板原图_{i:02d}.png", img)
        ok, centers = detect_grid(img, pattern_size)
        draw_detection(folder / f"相机内参标定_圆点检测_{i:02d}.png", img, pattern_size, centers, ok)
        if ok and centers is not None:
            obj_list.append(obj.copy())
            img_list.append(centers.astype(np.float32))
            saved.append((img, centers))
    if len(obj_list) < 8:
        raise RuntimeError("相机标定检测成功图像不足。")
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list,
        img_list,
        (true_camera.width, true_camera.height),
        None,
        None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    )
    for i, ((img, centers), rvec, tvec) in enumerate(zip(saved, rvecs, tvecs), 1):
        reproj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        draw_reprojection(folder / f"相机内参标定_重投影_{i:02d}.png", img, centers, reproj)
    cam = Camera(true_camera.width, true_camera.height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
    return cam, dist, float(rms), len(obj_list)


def calibrate_reference_plane(
    true_camera: Camera,
    calibrated_camera: Camera,
    dist: np.ndarray,
    reference: PlaneSurface,
    folder: Path,
) -> tuple[PlaneSurface, np.ndarray, float]:
    pattern_size = (9, 7)
    obj = make_grid_points(pattern_size, 0.035)
    rvec_true = np.zeros(3)
    tvec_true = np.array([0.0, 0.0, reference.z0])
    img, _ = render_circle_board(true_camera, obj, rvec_true, tvec_true, 5, 100)
    imwrite_unicode(folder / "参考平面标定_标定板放镜面原图.png", img)
    ok, centers = detect_grid(img, pattern_size)
    draw_detection(folder / "参考平面标定_圆点检测.png", img, pattern_size, centers, ok)
    if not ok or centers is None:
        raise RuntimeError("参考平面标定圆点检测失败。")
    ok_pnp, rvec, tvec = cv2.solvePnP(obj, centers, calibrated_camera.k(), dist)
    if not ok_pnp:
        raise RuntimeError("参考平面 solvePnP 失败。")
    reproj, _ = cv2.projectPoints(obj, rvec, tvec, calibrated_camera.k(), dist)
    draw_reprojection(folder / "参考平面标定_重投影.png", img, centers, reproj)
    r, _ = cv2.Rodrigues(rvec)
    n = r @ np.array([0.0, 0.0, 1.0])
    n = n / (np.linalg.norm(n) + EPS)
    d = -float(n @ tvec.reshape(3))
    plane = PlaneSurface(z0=float(tvec.reshape(3)[2]), name="标定参考平面")
    err = float(np.sqrt(np.mean(np.sum((centers.reshape(-1, 2) - reproj.reshape(-1, 2)) ** 2, axis=1))))
    return plane, np.r_[n, d], err


def save_lcd_calib_pattern(path: Path, screen: Screen, obj: np.ndarray) -> None:
    preview = np.full((540, 960), 245, dtype=np.uint8)
    for p in obj:
        uv = screen.world_to_uv(np.array([p[0], p[1], screen.z]))
        x = int(round(uv[0] / screen.width_px * preview.shape[1]))
        y = int(round(uv[1] / screen.height_px * preview.shape[0]))
        if 0 <= x < preview.shape[1] and 0 <= y < preview.shape[0]:
            cv2.circle(preview, (x, y), 8, 0, -1, lineType=cv2.LINE_AA)
    imwrite_unicode(path, preview)


def calibrate_screen_by_reflection(
    true_camera: Camera,
    calibrated_camera: Camera,
    dist: np.ndarray,
    true_screen: Screen,
    true_reference: PlaneSurface,
    calibrated_reference: PlaneSurface,
    folder: Path,
) -> tuple[Screen, float]:
    pattern_size = (11, 7)
    obj = make_grid_points(pattern_size, 0.09)
    save_lcd_calib_pattern(folder / "LCD屏幕外参标定_屏幕圆点图案.png", true_screen, obj)
    z_virtual = 2.0 * true_reference.z0 - true_screen.z
    img, _ = render_circle_board(true_camera, obj, np.zeros(3), np.array([0.0, 0.0, z_virtual]), 5, 200)
    imwrite_unicode(folder / "LCD屏幕外参标定_经参考镜反射原图.png", img)
    ok, centers = detect_grid(img, pattern_size)
    draw_detection(folder / "LCD屏幕外参标定_反射圆点检测.png", img, pattern_size, centers, ok)
    if not ok or centers is None:
        raise RuntimeError("屏幕外参标定圆点检测失败。")
    ok_pnp, rvec, tvec = cv2.solvePnP(obj, centers, calibrated_camera.k(), dist)
    if not ok_pnp:
        raise RuntimeError("屏幕虚像 solvePnP 失败。")
    reproj, _ = cv2.projectPoints(obj, rvec, tvec, calibrated_camera.k(), dist)
    draw_reprojection(folder / "LCD屏幕外参标定_虚像重投影.png", img, centers, reproj)
    screen_z = 2.0 * calibrated_reference.z0 - float(tvec.reshape(3)[2])
    screen = Screen(screen_z, true_screen.width_m, true_screen.height_m, true_screen.width_px, true_screen.height_px)
    err = float(np.sqrt(np.mean(np.sum((centers.reshape(-1, 2) - reproj.reshape(-1, 2)) ** 2, axis=1))))
    return screen, err


def apply_camera_effects(img: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vignette = 1.0 - 0.16 * rr
    out = img * vignette + 8.0
    out += rng.normal(0.0, 1.4, img.shape)
    out[~valid] = 0.0
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.float32), (3, 3), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def raytrace_uv(camera: Camera, screen: Screen, surface: Surface) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rays = camera.rays()
    p = surface.intersect(rays)
    n = surface.normal(p)
    reflected = reflect(rays, n)
    q, valid = screen.intersect(p, reflected)
    uv = screen.world_to_uv(q)
    return uv, p, q, valid


def sample_sinusoid(uv: np.ndarray, direction: str, period: float, step: float) -> np.ndarray:
    coord = uv[..., 0] if direction == "x" else uv[..., 1]
    return 128.0 + 88.0 * np.cos(2.0 * np.pi * coord / period + step)


def gray_code(value: np.ndarray) -> np.ndarray:
    return value ^ (value >> 1)


def gray_to_binary(gray: np.ndarray) -> np.ndarray:
    out = gray.copy()
    shift = 1
    while True:
        shifted = out >> shift
        if not np.any(shifted):
            break
        out ^= shifted
        shift <<= 1
    return out


def sample_gray(uv: np.ndarray, direction: str, period: float, bit: int, bits: int, n_periods: int) -> np.ndarray:
    coord = uv[..., 0] if direction == "x" else uv[..., 1]
    idx = np.clip(np.floor(coord / period).astype(np.int32), 0, n_periods - 1)
    g = gray_code(idx)
    value = (g >> (bits - bit - 1)) & 1
    return np.where(value > 0, 220.0, 35.0)


def save_screen_patterns(folder: Path, screen: Screen, period_x: float, period_y: float) -> None:
    yy, xx = np.mgrid[0 : screen.height_px, 0 : screen.width_px]
    for direction, period, coord in [("X", period_x, xx), ("Y", period_y, yy)]:
        for i, step in enumerate([0, 0.5 * np.pi, np.pi, 1.5 * np.pi], 1):
            img = 128 + 88 * np.cos(2 * np.pi * coord / period + step)
            preview = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
            imwrite_unicode(folder / f"{direction}方向四步相移屏幕图案_第{i}步.png", preview)
    for direction, period, total, coord in [
        ("X", period_x, screen.width_px, xx),
        ("Y", period_y, screen.height_px, yy),
    ]:
        n_periods = int(np.ceil(total / period))
        bits = int(np.ceil(np.log2(n_periods)))
        for bit in range(bits):
            idx = np.clip(np.floor(coord / period).astype(np.int32), 0, n_periods - 1)
            g = gray_code(idx)
            value = (g >> (bits - bit - 1)) & 1
            img = np.where(value > 0, 220, 35).astype(np.uint8)
            preview = cv2.resize(img, (960, 540), interpolation=cv2.INTER_NEAREST)
            imwrite_unicode(folder / f"{direction}方向GrayCode屏幕图案_第{bit + 1}位.png", preview)


def acquire_phase_images(
    camera: Camera,
    screen: Screen,
    surface: Surface,
    folder: Path,
    prefix: str,
    period_x: float,
    period_y: float,
    seed_base: int,
) -> dict[str, np.ndarray]:
    uv, p, q, valid = raytrace_uv(camera, screen, surface)
    steps = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    x_imgs = []
    y_imgs = []
    for i, step in enumerate(steps, 1):
        img = apply_camera_effects(sample_sinusoid(uv, "x", period_x, step), valid, seed_base + i)
        imwrite_unicode(folder / f"{prefix}_X方向相移采集图_第{i}步.png", img)
        x_imgs.append(img.astype(np.float64))
    for i, step in enumerate(steps, 1):
        img = apply_camera_effects(sample_sinusoid(uv, "y", period_y, step), valid, seed_base + 10 + i)
        imwrite_unicode(folder / f"{prefix}_Y方向相移采集图_第{i}步.png", img)
        y_imgs.append(img.astype(np.float64))

    gray_x = []
    gray_y = []
    nx = int(np.ceil(screen.width_px / period_x))
    bx = int(np.ceil(np.log2(nx)))
    ny = int(np.ceil(screen.height_px / period_y))
    by = int(np.ceil(np.log2(ny)))
    for bit in range(bx):
        img = apply_camera_effects(sample_gray(uv, "x", period_x, bit, bx, nx), valid, seed_base + 30 + bit)
        imwrite_unicode(folder / f"{prefix}_X方向GrayCode采集图_第{bit + 1}位.png", img)
        gray_x.append(img.astype(np.float64))
    for bit in range(by):
        img = apply_camera_effects(sample_gray(uv, "y", period_y, bit, by, ny), valid, seed_base + 50 + bit)
        imwrite_unicode(folder / f"{prefix}_Y方向GrayCode采集图_第{bit + 1}位.png", img)
        gray_y.append(img.astype(np.float64))

    return {
        "uv_true": uv,
        "points": p,
        "screen_points": q,
        "valid": valid,
        "x_imgs": np.stack(x_imgs),
        "y_imgs": np.stack(y_imgs),
        "gray_x": np.stack(gray_x),
        "gray_y": np.stack(gray_y),
    }


def four_step_phase(imgs: np.ndarray) -> np.ndarray:
    i1, i2, i3, i4 = imgs
    return np.arctan2(i4 - i2, i1 - i3)


def decode_absolute_phase(phase_imgs: np.ndarray, gray_imgs: np.ndarray, period: float, total_px: int) -> tuple[np.ndarray, np.ndarray]:
    wrapped = four_step_phase(phase_imgs)
    phase_mod = np.mod(wrapped, 2.0 * np.pi)
    bits = gray_imgs.shape[0]
    gray = np.zeros_like(phase_mod, dtype=np.int32)
    for bit in range(bits):
        b = (gray_imgs[bit] > 128).astype(np.int32)
        gray += b << (bits - bit - 1)
    order = gray_to_binary(gray)
    order = np.clip(order, 0, int(np.ceil(total_px / period)) - 1)
    absolute = order * 2.0 * np.pi + phase_mod
    return wrapped, absolute


def reference_distance_mean(
    camera: Camera,
    screen: Screen,
    reference: PlaneSurface,
    abs_x: np.ndarray,
    abs_y: np.ndarray,
    period_x: float,
    period_y: float,
    valid: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    rays = camera.rays()
    p = reference.intersect(rays)
    uv = np.stack([period_x * abs_x / (2.0 * np.pi), period_y * abs_y / (2.0 * np.pi)], axis=-1)
    q = screen.uv_to_world(uv)
    dist = np.linalg.norm(q - p, axis=-1)
    return float(np.mean(dist[valid])), p, uv


def integrate_poisson_dct(p: np.ndarray, q: np.ndarray, dx: float, dy: float) -> np.ndarray:
    div = np.gradient(p, dx, axis=1, edge_order=2) + np.gradient(q, dy, axis=0, edge_order=2)
    h, w = div.shape
    f_hat = dctn(div, type=2, norm="ortho")
    ky = np.arange(h)
    kx = np.arange(w)
    lam_y = 2 * (np.cos(np.pi * ky / h) - 1) / (dy * dy)
    lam_x = 2 * (np.cos(np.pi * kx / w) - 1) / (dx * dx)
    denom = lam_y[:, None] + lam_x[None, :]
    z_hat = np.zeros_like(f_hat)
    ok = denom != 0
    z_hat[ok] = f_hat[ok] / denom[ok]
    z = idctn(z_hat, type=2, norm="ortho")
    return z - np.mean(z)


def align_mean(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return z - np.mean(z[mask])


def save_scalar(path: Path, data: np.ndarray, title: str, unit: str, cmap: str, mask: np.ndarray | None = None) -> None:
    arr = np.asarray(data, dtype=np.float64)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    plt.figure(figsize=(7.2, 5.2), dpi=180)
    im = plt.imshow(arr, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    if unit:
        cb.set_label(unit)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_surface(
    path: Path,
    z: np.ndarray,
    title: str,
    zlabel: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    cmap: str = "jet",
) -> None:
    data = np.asarray(z, dtype=np.float64)
    h, w = data.shape
    if x is None or y is None:
        yy, xx = np.mgrid[0:h, 0:w]
    else:
        xx, yy = x, y
    if mask is not None:
        data = np.where(mask, data, np.nan)
    stride = max(1, min(h, w) // 180)
    xs = xx[::stride, ::stride]
    ys = yy[::stride, ::stride]
    zs = data[::stride, ::stride]
    fig = plt.figure(figsize=(8.8, 6.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xs, ys, np.ma.masked_invalid(zs), cmap=cmap, linewidth=0, antialiased=True)
    finite = zs[np.isfinite(zs)]
    if finite.size:
        zmin = float(np.nanmin(finite))
        zmax = float(np.nanmax(finite))
        ax.contourf(xs, ys, zs, zdir="z", offset=zmin, levels=18, cmap=cmap, alpha=0.72)
        ax.set_zlim(zmin, zmax)
    ax.set_title(title)
    ax.set_xlabel("X/mm")
    ax.set_ylabel("Y/mm")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=26, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def save_system_schematic_3d(path: Path, camera: Camera, screen: Screen, reference: PlaneSurface) -> None:
    fig = plt.figure(figsize=(9.0, 6.5), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    sx = np.linspace(-screen.width_m / 2, screen.width_m / 2, 2)
    sy = np.linspace(-screen.height_m / 2, screen.height_m / 2, 2)
    sxg, syg = np.meshgrid(sx, sy)
    szg = np.full_like(sxg, screen.z)
    ax.plot_surface(sxg, syg, szg, color="#4C78A8", alpha=0.28, linewidth=0)
    mx = np.linspace(-0.48, 0.48, 2)
    my = np.linspace(-0.34, 0.34, 2)
    mxg, myg = np.meshgrid(mx, my)
    mzg = np.full_like(mxg, reference.z0)
    ax.plot_surface(mxg, myg, mzg, color="#59A14F", alpha=0.35, linewidth=0)
    ax.scatter([0], [0], [0], c="#E45756", s=60, label="相机中心")
    for x, y in [(-0.18, -0.12), (0.0, 0.0), (0.18, 0.12)]:
        p = np.array([x, y, reference.z0])
        n = np.array([0.0, 0.0, 1.0])
        d = normalize(p)
        r = reflect(d, n)
        t = (screen.z - p[2]) / (r[2] + EPS)
        q = p + t * r
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color="#E45756", lw=1.6)
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#F28E2B", lw=1.6)
        ax.scatter([p[0]], [p[1]], [p[2]], c="#59A14F", s=18)
        ax.scatter([q[0]], [q[1]], [q[2]], c="#4C78A8", s=18)
    ax.text(0, 0, 0, "Camera", color="#E45756")
    ax.text(-0.85, -0.55, screen.z, "LCD screen", color="#4C78A8")
    ax.text(-0.45, -0.32, reference.z0, "Reference mirror", color="#59A14F")
    ax.set_title("单目 PMD 实验系统三维示意图")
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    ax.view_init(elev=23, azim=-52)
    ax.set_box_aspect((1.8, 1.2, 1.0))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def process_object(
    obj_surface: Surface,
    true_camera: Camera,
    true_screen: Screen,
    calibrated_camera: Camera,
    calibrated_screen: Screen,
    calibrated_reference: PlaneSurface,
    reference_phase: dict[str, np.ndarray],
    ra_mean: float,
    ref_points: np.ndarray,
    valid_reference: np.ndarray,
    period_x: float,
    period_y: float,
    folder: Path,
    seed: int,
) -> dict[str, float]:
    folder.mkdir(parents=True, exist_ok=True)
    acq = acquire_phase_images(true_camera, true_screen, obj_surface, folder / "01_被测相位采集图像", obj_surface.name, period_x, period_y, seed)
    wrap_x, abs_x = decode_absolute_phase(acq["x_imgs"], acq["gray_x"], period_x, true_screen.width_px)
    wrap_y, abs_y = decode_absolute_phase(acq["y_imgs"], acq["gray_y"], period_y, true_screen.height_px)
    valid = valid_reference & acq["valid"]

    dphi_x = wrap_to_pi(abs_x - reference_phase["abs_x"])
    dphi_y = wrap_to_pi(abs_y - reference_phase["abs_y"])

    # 由用户给出的简化公式反推斜率。符号取决于屏幕坐标、相机坐标和条纹方向定义；
    # 本仿真中的坐标定义下直接使用正号。真实实验可用已知凸/凹标准件校准符号。
    grad_x = dphi_x * period_x * calibrated_screen.pp_x / (4.0 * np.pi * ra_mean)
    grad_y = dphi_y * period_y * calibrated_screen.pp_y / (4.0 * np.pi * ra_mean)
    grad_x = np.where(valid, grad_x, 0.0)
    grad_y = np.where(valid, grad_y, 0.0)

    dx = calibrated_reference.z0 / calibrated_camera.fx
    dy = calibrated_reference.z0 / calibrated_camera.fy
    z_rec = integrate_poisson_dct(grad_x, grad_y, dx, dy)
    z_true = obj_surface.height_xy(ref_points[..., 0], ref_points[..., 1])
    z_true = align_mean(z_true, valid)
    z_rec = align_mean(z_rec, valid)
    z_err = np.where(valid, z_rec - z_true, np.nan)

    phase_dir = folder / "02_相位解算结果"
    diff_dir = folder / "03_相位差结果"
    grad_dir = folder / "04_梯度结果"
    recon_dir = folder / "05_面型重建结果"
    for d in [phase_dir, diff_dir, grad_dir, recon_dir]:
        d.mkdir(parents=True, exist_ok=True)

    save_scalar(phase_dir / "X方向包裹相位图.png", wrap_x, "X方向包裹相位", "rad", "twilight", valid)
    save_scalar(phase_dir / "Y方向包裹相位图.png", wrap_y, "Y方向包裹相位", "rad", "twilight", valid)
    save_scalar(phase_dir / "X方向解包裹绝对相位图.png", abs_x, "X方向解包裹绝对相位", "rad", "jet", valid)
    save_scalar(phase_dir / "Y方向解包裹绝对相位图.png", abs_y, "Y方向解包裹绝对相位", "rad", "jet", valid)
    save_surface(phase_dir / "X方向包裹相位三维图.png", wrap_x, "X方向包裹相位三维图", "phase/rad", mask=valid, cmap="jet")
    save_surface(phase_dir / "X方向解包裹绝对相位三维图.png", abs_x, "X方向解包裹绝对相位三维图", "Absolute phase/rad", mask=valid, cmap="jet")

    save_scalar(diff_dir / "X方向相位差图.png", dphi_x, "X方向相位差", "rad", "coolwarm", valid)
    save_scalar(diff_dir / "Y方向相位差图.png", dphi_y, "Y方向相位差", "rad", "coolwarm", valid)
    save_surface(diff_dir / "X方向相位差三维图.png", dphi_x, "X方向相位差三维图", "phase/rad", mask=valid, cmap="coolwarm")
    save_surface(diff_dir / "Y方向相位差三维图.png", dphi_y, "Y方向相位差三维图", "phase/rad", mask=valid, cmap="coolwarm")

    save_scalar(grad_dir / "X方向梯度图.png", grad_x, "X方向梯度", "", "coolwarm", valid)
    save_scalar(grad_dir / "Y方向梯度图.png", grad_y, "Y方向梯度", "", "coolwarm", valid)
    save_surface(grad_dir / "X方向梯度三维图.png", grad_x, "X方向梯度三维图", "dz/dx", mask=valid, cmap="coolwarm")
    save_surface(grad_dir / "Y方向梯度三维图.png", grad_y, "Y方向梯度三维图", "dz/dy", mask=valid, cmap="coolwarm")

    x_mm = ref_points[..., 0] * 1e3
    y_mm = ref_points[..., 1] * 1e3
    save_scalar(recon_dir / "真实高度图.png", z_true * 1e6, "真实高度图", "um", "viridis", valid)
    save_scalar(recon_dir / "重建高度图.png", z_rec * 1e6, "重建高度图", "um", "viridis", valid)
    save_scalar(recon_dir / "高度误差图.png", z_err * 1e6, "高度误差图", "um", "coolwarm", valid)
    save_surface(recon_dir / "真实高度三维面型图.png", z_true * 1e3, "真实高度三维面型图", "Z/mm", x_mm, y_mm, valid, "jet")
    save_surface(recon_dir / "重建高度三维面型图.png", z_rec * 1e3, "重建高度三维面型图", "Z/mm", x_mm, y_mm, valid, "jet")
    save_surface(recon_dir / "高度误差三维图.png", z_err * 1e6, "高度误差三维图", "error/um", x_mm, y_mm, valid, "coolwarm")

    rmse = float(np.sqrt(np.nanmean(z_err[valid] ** 2)))
    mae = float(np.nanmean(np.abs(z_err[valid])))
    max_err = float(np.nanmax(np.abs(z_err[valid])))
    pv_true = float(np.nanmax(z_true[valid]) - np.nanmin(z_true[valid]))
    pv_rec = float(np.nanmax(z_rec[valid]) - np.nanmin(z_rec[valid]))
    return {
        "rmse_um": rmse * 1e6,
        "mae_um": mae * 1e6,
        "max_error_um": max_err * 1e6,
        "true_pv_um": pv_true * 1e6,
        "rec_pv_um": pv_rec * 1e6,
    }


def main() -> None:
    root = Path("results") / "10_真实实验流程仿真"
    if root.exists():
        shutil.rmtree(root)
    dirs = {
        "camera": root / "01_相机内参标定",
        "reference": root / "02_参考平面镜标定",
        "screen": root / "03_LCD屏幕外参标定",
        "patterns": root / "04_屏幕投射图案",
        "ref_phase": root / "05_平面镜参考相位",
        "objects": root / "06_被测物重建结果",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    true_camera = Camera()
    true_screen = Screen()
    true_reference = PlaneSurface(1.0, "标准平面镜")
    period_x = 96.0
    period_y = 90.0

    save_system_schematic_3d(root / "PMD实验系统三维示意图.png", true_camera, true_screen, true_reference)

    calibrated_camera, dist, cam_rms, cam_count = calibrate_camera_from_synthetic_images(true_camera, dirs["camera"])
    calibrated_reference, ref_plane, ref_err = calibrate_reference_plane(
        true_camera, calibrated_camera, dist, true_reference, dirs["reference"]
    )
    calibrated_screen, screen_err = calibrate_screen_by_reflection(
        true_camera,
        calibrated_camera,
        dist,
        true_screen,
        true_reference,
        calibrated_reference,
        dirs["screen"],
    )

    save_screen_patterns(dirs["patterns"], true_screen, period_x, period_y)

    ref_acq = acquire_phase_images(true_camera, true_screen, true_reference, dirs["ref_phase"] / "01_参考相位采集图像", "参考平面镜", period_x, period_y, 500)
    ref_wrap_x, ref_abs_x = decode_absolute_phase(ref_acq["x_imgs"], ref_acq["gray_x"], period_x, true_screen.width_px)
    ref_wrap_y, ref_abs_y = decode_absolute_phase(ref_acq["y_imgs"], ref_acq["gray_y"], period_y, true_screen.height_px)
    ref_valid = ref_acq["valid"]
    ref_phase = {"wrap_x": ref_wrap_x, "wrap_y": ref_wrap_y, "abs_x": ref_abs_x, "abs_y": ref_abs_y}

    phase_dir = dirs["ref_phase"] / "02_参考相位解算结果"
    phase_dir.mkdir(parents=True, exist_ok=True)
    save_scalar(phase_dir / "X方向参考包裹相位图.png", ref_wrap_x, "X方向参考包裹相位", "rad", "twilight", ref_valid)
    save_scalar(phase_dir / "Y方向参考包裹相位图.png", ref_wrap_y, "Y方向参考包裹相位", "rad", "twilight", ref_valid)
    save_scalar(phase_dir / "X方向参考解包裹绝对相位图.png", ref_abs_x, "X方向参考解包裹绝对相位", "rad", "jet", ref_valid)
    save_scalar(phase_dir / "Y方向参考解包裹绝对相位图.png", ref_abs_y, "Y方向参考解包裹绝对相位", "rad", "jet", ref_valid)
    save_surface(phase_dir / "X方向参考包裹相位三维图.png", ref_wrap_x, "X方向参考包裹相位三维图", "phase/rad", mask=ref_valid, cmap="jet")
    save_surface(phase_dir / "X方向参考解包裹绝对相位三维图.png", ref_abs_x, "X方向参考解包裹绝对相位三维图", "Absolute phase/rad", mask=ref_valid, cmap="jet")

    ra_mean, ref_points, ref_uv = reference_distance_mean(
        calibrated_camera,
        calibrated_screen,
        calibrated_reference,
        ref_abs_x,
        ref_abs_y,
        period_x,
        period_y,
        ref_valid,
    )

    objects: list[Surface] = [
        GaussianDefectSurface(1.0, amp=1.2e-3, sigma=0.075, x0=0.04, y0=-0.03, name="带局部高斯凸包的平板"),
        GaussianDefectSurface(1.0, amp=-1.0e-3, sigma=0.085, x0=-0.05, y0=0.04, name="带局部高斯凹坑的平板"),
        SmoothSemiDomeSurface(1.0, amp=0.9e-3, aperture=0.18, x0=0.0, y0=0.0, name="局部平滑半圆帽面"),
    ]
    metrics: dict[str, dict[str, float]] = {}
    for i, obj in enumerate(objects, 1):
        obj_folder = dirs["objects"] / f"{i:02d}_{obj.name}"
        metrics[obj.name] = process_object(
            obj,
            true_camera,
            true_screen,
            calibrated_camera,
            calibrated_screen,
            calibrated_reference,
            ref_phase,
            ra_mean,
            ref_points,
            ref_valid,
            period_x,
            period_y,
            obj_folder,
            seed=800 + i * 100,
        )

    lines = [
        "真实实验流程式单目 PMD 仿真 summary",
        "",
        "【流程可行性判断】",
        "该流程可以作为单目 PMD 验证流程：相机内参、参考平面、LCD屏幕外参、参考相位、被测相位、相位差、斜率积分均被显式模拟。",
        "第六步使用平均 RA 的简化公式，适合小斜率、小高度变化、参考镜与被测镜位置近似一致的验证实验。",
        "若被测面高度变化大、斜率大或离轴几何明显，应改用逐像素 C-P-Q 反射几何求法向。",
        "",
        "【标定指标】",
        f"相机标定成功图像数 = {cam_count}",
        f"相机内参标定 RMS = {cam_rms:.6f} px",
        f"真实相机 fx, fy = {true_camera.fx:.3f}, {true_camera.fy:.3f} px",
        f"标定相机 fx, fy = {calibrated_camera.fx:.3f}, {calibrated_camera.fy:.3f} px",
        f"参考平面 PnP 重投影误差 = {ref_err:.6f} px",
        f"参考平面方程 n_x,n_y,n_z,d = {ref_plane[0]:.9f}, {ref_plane[1]:.9f}, {ref_plane[2]:.9f}, {ref_plane[3]:.9f}",
        f"LCD屏幕虚像 PnP 重投影误差 = {screen_err:.6f} px",
        f"LCD像素物理尺寸 pp_x = {calibrated_screen.pp_x:.9e} m/px",
        f"LCD像素物理尺寸 pp_y = {calibrated_screen.pp_y:.9e} m/px",
        f"平均参考距离 mean(RA) = {ra_mean:.9f} m",
        f"条纹周期 T_x = {period_x:.3f} px",
        f"条纹周期 T_y = {period_y:.3f} px",
        "",
        "【被测物重建误差】",
    ]
    for name, m in metrics.items():
        lines.extend(
            [
                f"{name}:",
                f"  真实高度 PV = {m['true_pv_um']:.6f} um",
                f"  重建高度 PV = {m['rec_pv_um']:.6f} um",
                f"  RMSE = {m['rmse_um']:.6f} um",
                f"  MAE = {m['mae_um']:.6f} um",
                f"  最大误差 = {m['max_error_um']:.6f} um",
            ]
        )
    (root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"结果已保存到：{root.resolve()}")
    print(f"mean(RA) = {ra_mean:.6f} m")
    for name, m in metrics.items():
        print(f"{name}: RMSE={m['rmse_um']:.3f} um, MAE={m['mae_um']:.3f} um, Max={m['max_error_um']:.3f} um")


if __name__ == "__main__":
    main()
