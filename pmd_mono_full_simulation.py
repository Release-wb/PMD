"""
单目相位偏折术 PMD 全仿真验证程序。

本程序不连接真实相机和 LCD 屏幕，而是先设定一组真实的
相机-屏幕-参考平面几何参数，再用光线追迹和镜面反射定律生成
“相机实际会采集到的相移条纹图像”。随后从这些模拟图像出发，
完成四步相移相位解算、参考/被测相位差、梯度估计和 DCT-Poisson
面型积分重建。

依赖：
    numpy, opencv-python, matplotlib, scipy

运行：
    python pmd_mono_full_simulation.py
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
    width: int = 320
    height: int = 240
    fx: float = 360.0
    fy: float = 360.0
    cx: float = 159.5
    cy: float = 119.5

    def pixel_rays(self) -> np.ndarray:
        """根据相机内参，为每个像素生成一条归一化相机射线。"""
        v, u = np.mgrid[0 : self.height, 0 : self.width]
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        rays = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float64)
        return normalize(rays)


@dataclass
class Screen:
    """LCD 屏幕平面，这里简化为世界坐标系中的 z = z0 平面。"""

    z: float = 0.20
    width_m: float = 1.80
    height_m: float = 1.25
    width_px: int = 1920
    height_px: int = 1080

    @property
    def pitch_x(self) -> float:
        return self.width_m / self.width_px

    @property
    def pitch_y(self) -> float:
        return self.height_m / self.height_px

    def intersect(self, points: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """求反射光线 P + t*d 与屏幕平面的交点。"""
        t = (self.z - points[..., 2]) / (directions[..., 2] + EPS)
        hit = points + t[..., None] * directions
        valid = t > 0
        return hit, valid

    def world_to_uv(self, points: np.ndarray) -> np.ndarray:
        """把屏幕平面上的三维点转换为 LCD 像素坐标。"""
        u = (points[..., 0] + 0.5 * self.width_m) / self.pitch_x
        v = (points[..., 1] + 0.5 * self.height_m) / self.pitch_y
        return np.stack([u, v], axis=-1)

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        """把 LCD 像素坐标转换为屏幕平面上的三维点。"""
        x = uv[..., 0] * self.pitch_x - 0.5 * self.width_m
        y = uv[..., 1] * self.pitch_y - 0.5 * self.height_m
        z = np.full_like(x, self.z, dtype=np.float64)
        return np.stack([x, y, z], axis=-1)

    def inside(self, uv: np.ndarray) -> np.ndarray:
        return (
            (uv[..., 0] >= 0)
            & (uv[..., 0] <= self.width_px - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= self.height_px - 1)
        )


@dataclass
class GaussianMirror:
    """被测高光面：参考平面上叠加一个高斯凸包或凹坑。"""

    z0: float = 1.0
    amplitude: float = 0.0
    sigma: float = 0.09
    x0: float = 0.03
    y0: float = -0.02

    def height_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2
        return self.amplitude * np.exp(-r2 / (2.0 * self.sigma**2))

    def grad_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.height_xy(x, y)
        hx = -(x - self.x0) / (self.sigma**2) * h
        hy = -(y - self.y0) / (self.sigma**2) * h
        return hx, hy

    def intersect(self, rays: np.ndarray, iterations: int = 8) -> np.ndarray:
        """
        求相机射线 C + t*d 与镜面 z = z0 + h(x,y) 的交点。

        Newton 迭代求解：
            t*dz = z0 + h(t*dx, t*dy)
        """
        dz = rays[..., 2]
        t = self.z0 / (dz + EPS)
        for _ in range(iterations):
            x = t * rays[..., 0]
            y = t * rays[..., 1]
            h = self.height_xy(x, y)
            hx, hy = self.grad_xy(x, y)
            f = t * dz - self.z0 - h
            df = dz - hx * rays[..., 0] - hy * rays[..., 1]
            t -= f / (df + EPS)
        return t[..., None] * rays

    def normal(self, points: np.ndarray) -> np.ndarray:
        hx, hy = self.grad_xy(points[..., 0], points[..., 1])
        n = np.stack([-hx, -hy, np.ones_like(hx)], axis=-1)
        return normalize(n)


def camera_matrix(camera: Camera) -> np.ndarray:
    """把 Camera 数据类转换为 OpenCV 相机内参矩阵。"""
    return np.array(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def camera_from_matrix(k: np.ndarray, width: int, height: int) -> Camera:
    """把 OpenCV 标定得到的内参矩阵转换回 Camera 数据类。"""
    return Camera(
        width=width,
        height=height,
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
    )


def make_symmetric_circle_grid(pattern_size: tuple[int, int], spacing: float) -> np.ndarray:
    """生成对称圆点标定板三维点，坐标原点放在圆点阵列中心。"""
    cols, rows = pattern_size
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j - (cols - 1) / 2.0) * spacing
            y = (i - (rows - 1) / 2.0) * spacing
            points.append([x, y, 0.0])
    return np.asarray(points, dtype=np.float32)


def make_blob_detector() -> cv2.SimpleBlobDetector:
    """为黑色圆点板配置 OpenCV blob 检测器。"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 1200
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(params)


def render_projected_circle_grid(
    camera: Camera,
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    radius_px: int = 6,
    noise_std: float = 0.8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    模拟相机拍摄圆点标定板。

    这里先把三维圆点中心投影到相机像面，再绘制成黑色圆点；
    这等价于生成一张可被 OpenCV 检测的标定板采集图像。
    """
    rng = np.random.default_rng(seed)
    k = camera_matrix(camera)
    dist = np.zeros(5)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, k, dist)
    image_points = image_points.reshape(-1, 2)

    img = np.full((camera.height, camera.width), 245, dtype=np.float64)
    for p in image_points:
        if 0 <= p[0] < camera.width and 0 <= p[1] < camera.height:
            cv2.circle(img, tuple(np.round(p).astype(int)), radius_px, 0, -1, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img += rng.normal(0.0, noise_std, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), image_points


def detect_circle_grid(
    image: np.ndarray,
    pattern_size: tuple[int, int],
    blob_detector: cv2.SimpleBlobDetector,
) -> tuple[bool, np.ndarray | None]:
    """从模拟采集图像中检测圆点阵列中心。"""
    flags = cv2.CALIB_CB_SYMMETRIC_GRID
    ok, centers = cv2.findCirclesGrid(image, pattern_size, flags=flags, blobDetector=blob_detector)
    return ok, centers


def save_detection_overlay(
    path: Path,
    image: np.ndarray,
    pattern_size: tuple[int, int],
    centers: np.ndarray | None,
    ok: bool,
) -> None:
    """保存圆点检测过程图。"""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if centers is not None:
        cv2.drawChessboardCorners(vis, pattern_size, centers, ok)
    imwrite_unicode(path, vis)


def save_reprojection_overlay(
    path: Path,
    image: np.ndarray,
    detected_centers: np.ndarray,
    reprojected_points: np.ndarray,
) -> None:
    """保存检测点和重投影点叠加图，用于检查标定质量。"""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detected = detected_centers.reshape(-1, 2)
    reproj = reprojected_points.reshape(-1, 2)
    for p in detected:
        cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0, 180, 0), -1, lineType=cv2.LINE_AA)
    for p in reproj:
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


def save_lcd_dot_pattern(path: Path, screen: Screen, object_points: np.ndarray) -> None:
    """保存 LCD 上用于屏幕标定的圆点阵列图案预览。"""
    preview_w, preview_h = 960, 540
    preview = np.full((preview_h, preview_w), 245, dtype=np.uint8)
    for p in object_points:
        uv = screen.world_to_uv(np.array([p[0], p[1], screen.z], dtype=np.float64))
        x = int(round(uv[0] / screen.width_px * preview_w))
        y = int(round(uv[1] / screen.height_px * preview_h))
        if 0 <= x < preview_w and 0 <= y < preview_h:
            cv2.circle(preview, (x, y), 9, 0, -1, lineType=cv2.LINE_AA)
    imwrite_unicode(path, preview)


def run_camera_calibration(
    true_camera: Camera,
    folder: Path,
) -> tuple[Camera, np.ndarray, float, int]:
    """模拟多姿态圆点板采集，并用 OpenCV calibrateCamera 标定相机内参。"""
    folder.mkdir(parents=True, exist_ok=True)
    pattern_size = (9, 7)
    object_points = make_symmetric_circle_grid(pattern_size, spacing=0.035)
    blob_detector = make_blob_detector()

    poses = [
        ([0.00, 0.00, 0.00], [0.00, 0.00, 0.75]),
        ([0.18, -0.12, 0.08], [-0.05, -0.03, 0.78]),
        ([-0.16, 0.10, -0.10], [0.05, 0.03, 0.82]),
        ([0.12, 0.18, 0.14], [-0.04, 0.04, 0.86]),
        ([-0.20, -0.14, 0.10], [0.05, -0.04, 0.90]),
        ([0.08, -0.20, -0.16], [-0.03, 0.02, 0.70]),
        ([-0.12, 0.22, 0.04], [0.04, -0.02, 0.73]),
        ([0.22, 0.04, -0.12], [-0.06, 0.01, 0.88]),
        ([-0.08, -0.22, 0.18], [0.03, 0.05, 0.80]),
        ([0.15, 0.15, -0.18], [-0.02, -0.05, 0.84]),
        ([-0.22, 0.04, 0.16], [0.06, 0.02, 0.92]),
        ([0.05, -0.10, 0.22], [0.00, -0.04, 0.76]),
    ]

    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    saved_images: list[np.ndarray] = []
    saved_centers: list[np.ndarray] = []

    for idx, (rvec_raw, tvec_raw) in enumerate(poses, 1):
        rvec = np.asarray(rvec_raw, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec_raw, dtype=np.float64).reshape(3, 1)
        img, _ = render_projected_circle_grid(true_camera, object_points, rvec, tvec, radius_px=6, seed=idx)
        imwrite_unicode(folder / f"相机标定圆点板_第{idx:02d}张.png", img)
        ok, centers = detect_circle_grid(img, pattern_size, blob_detector)
        save_detection_overlay(folder / f"相机标定检测结果_第{idx:02d}张.png", img, pattern_size, centers, ok)
        if ok and centers is not None:
            object_points_list.append(object_points.copy())
            image_points_list.append(centers.astype(np.float32))
            saved_images.append(img)
            saved_centers.append(centers)

    if len(object_points_list) < 6:
        raise RuntimeError("相机标定圆点检测数量不足，请增大圆点半径或调整标定板姿态。")

    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, k_est, dist_est, rvecs_est, tvecs_est = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        (true_camera.width, true_camera.height),
        None,
        None,
        flags=flags,
    )

    for idx, (img, centers, rvec, tvec) in enumerate(zip(saved_images, saved_centers, rvecs_est, tvecs_est), 1):
        reproj, _ = cv2.projectPoints(object_points, rvec, tvec, k_est, dist_est)
        save_reprojection_overlay(folder / f"相机标定重投影结果_第{idx:02d}张.png", img, centers, reproj)

    return camera_from_matrix(k_est, true_camera.width, true_camera.height), dist_est, float(rms), len(object_points_list)


def run_reference_plane_calibration(
    true_camera: Camera,
    calibrated_camera: Camera,
    dist_est: np.ndarray,
    true_reference: GaussianMirror,
    folder: Path,
) -> tuple[GaussianMirror, float]:
    """模拟把圆点板放在参考平面位置，并用 solvePnP 估计参考平面位置。"""
    folder.mkdir(parents=True, exist_ok=True)
    pattern_size = (9, 7)
    object_points = make_symmetric_circle_grid(pattern_size, spacing=0.035)
    blob_detector = make_blob_detector()

    rvec_true = np.zeros((3, 1), dtype=np.float64)
    tvec_true = np.array([[0.0], [0.0], [true_reference.z0]], dtype=np.float64)
    img, _ = render_projected_circle_grid(true_camera, object_points, rvec_true, tvec_true, radius_px=5, seed=101)
    imwrite_unicode(folder / "参考平面圆点板模拟图像.png", img)

    ok, centers = detect_circle_grid(img, pattern_size, blob_detector)
    save_detection_overlay(folder / "参考平面圆点检测结果.png", img, pattern_size, centers, ok)
    if not ok or centers is None:
        raise RuntimeError("参考平面圆点检测失败。")

    k_est = camera_matrix(calibrated_camera)
    ok_pnp, rvec_est, tvec_est = cv2.solvePnP(object_points, centers, k_est, dist_est, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok_pnp:
        raise RuntimeError("参考平面 solvePnP 失败。")

    reproj, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, k_est, dist_est)
    save_reprojection_overlay(folder / "参考平面重投影结果.png", img, centers, reproj)

    reference_est = GaussianMirror(z0=float(tvec_est[2, 0]), amplitude=0.0)
    reproj_error = float(np.sqrt(np.mean(np.sum((centers.reshape(-1, 2) - reproj.reshape(-1, 2)) ** 2, axis=1))))
    return reference_est, reproj_error


def run_screen_calibration(
    true_camera: Camera,
    calibrated_camera: Camera,
    dist_est: np.ndarray,
    true_screen: Screen,
    true_reference: GaussianMirror,
    calibrated_reference: GaussianMirror,
    folder: Path,
) -> tuple[Screen, float]:
    """
    模拟 LCD 显示圆点阵列，经参考平面镜反射后被相机采集。

    对平面镜来说，相机看到的是真实屏幕关于参考平面的虚像。
    因此先用 solvePnP 估计虚拟屏幕平面，再由参考平面反射回真实屏幕位置。
    """
    folder.mkdir(parents=True, exist_ok=True)
    pattern_size = (11, 7)
    object_points = make_symmetric_circle_grid(pattern_size, spacing=0.09)
    blob_detector = make_blob_detector()

    save_lcd_dot_pattern(folder / "LCD屏幕圆点阵列图案.png", true_screen, object_points)

    z_virtual = 2.0 * true_reference.z0 - true_screen.z
    rvec_virtual = np.zeros((3, 1), dtype=np.float64)
    tvec_virtual = np.array([[0.0], [0.0], [z_virtual]], dtype=np.float64)
    img, _ = render_projected_circle_grid(true_camera, object_points, rvec_virtual, tvec_virtual, radius_px=5, seed=202)
    imwrite_unicode(folder / "屏幕经参考镜模拟采集图像.png", img)

    ok, centers = detect_circle_grid(img, pattern_size, blob_detector)
    save_detection_overlay(folder / "屏幕标定圆点检测结果.png", img, pattern_size, centers, ok)
    if not ok or centers is None:
        raise RuntimeError("屏幕标定圆点检测失败。")

    k_est = camera_matrix(calibrated_camera)
    ok_pnp, rvec_est, tvec_est = cv2.solvePnP(object_points, centers, k_est, dist_est, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok_pnp:
        raise RuntimeError("屏幕虚像 solvePnP 失败。")

    reproj, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, k_est, dist_est)
    save_reprojection_overlay(folder / "屏幕虚像重投影结果.png", img, centers, reproj)

    screen_z_est = 2.0 * calibrated_reference.z0 - float(tvec_est[2, 0])
    screen_est = Screen(
        z=screen_z_est,
        width_m=true_screen.width_m,
        height_m=true_screen.height_m,
        width_px=true_screen.width_px,
        height_px=true_screen.height_px,
    )
    reproj_error = float(np.sqrt(np.mean(np.sum((centers.reshape(-1, 2) - reproj.reshape(-1, 2)) ** 2, axis=1))))
    return screen_est, reproj_error


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def reflect(directions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """镜面反射：把入射方向按照表面法向反射出去。"""
    dot = np.sum(directions * normals, axis=-1, keepdims=True)
    return normalize(directions - 2.0 * dot * normals)


def wrap_to_pi(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def make_screen_patterns(
    screen: Screen,
    direction: str,
    period_px: float,
    phase_steps: list[float],
    background: float = 128.0,
    modulation: float = 95.0,
) -> list[np.ndarray]:
    """生成 LCD 上显示的 X/Y 方向四步相移正弦条纹。"""
    yy, xx = np.mgrid[0 : screen.height_px, 0 : screen.width_px]
    coord = xx if direction.lower() == "x" else yy
    patterns = []
    for step in phase_steps:
        img = background + modulation * np.cos(2.0 * np.pi * coord / period_px + step)
        patterns.append(np.clip(img, 0, 255).astype(np.uint8))
    return patterns


def sample_screen_intensity(
    uv: np.ndarray,
    direction: str,
    period_px: float,
    phase_step: float,
    background: float = 128.0,
    modulation: float = 95.0,
) -> np.ndarray:
    """在给定 LCD 坐标处采样连续正弦条纹灰度。"""
    coord = uv[..., 0] if direction.lower() == "x" else uv[..., 1]
    return background + modulation * np.cos(2.0 * np.pi * coord / period_px + phase_step)


def simulate_camera_images(
    camera: Camera,
    screen: Screen,
    surface: GaussianMirror,
    direction: str,
    period_px: float,
    phase_steps: list[float],
    noise_std: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按真实 PMD 光路模拟相机采集图像：
        相机像素射线 -> 镜面交点 -> 按法向反射 -> 屏幕条纹采样

    返回：
        images: 四步相移图像，形状为 (4, H, W)
        uv: 每个相机像素经镜面反射后看到的 LCD 坐标
        points: 相机射线打到镜面上的三维点
        valid: 是否有效看到屏幕区域
    """
    rng = np.random.default_rng(seed)
    rays = camera.pixel_rays()
    points = surface.intersect(rays)
    normals = surface.normal(points)
    reflected = reflect(rays, normals)
    screen_points, valid_t = screen.intersect(points, reflected)
    uv = screen.world_to_uv(screen_points)
    valid = valid_t & screen.inside(uv)

    images = []
    for step in phase_steps:
        img = sample_screen_intensity(uv, direction, period_px, step)
        img += rng.normal(0.0, noise_std, size=img.shape)
        img[~valid] = 0.0
        images.append(np.clip(img, 0, 255).astype(np.float64))
    return np.stack(images, axis=0), uv, points, valid


def compute_reference_uv_from_geometry(
    camera: Camera,
    screen: Screen,
    reference: GaussianMirror,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用标定得到的相机、屏幕和参考平面几何关系计算参考屏幕坐标。"""
    rays = camera.pixel_rays()
    points = reference.intersect(rays)
    normals = reference.normal(points)
    reflected = reflect(rays, normals)
    screen_points, valid_t = screen.intersect(points, reflected)
    uv = screen.world_to_uv(screen_points)
    return uv, points, valid_t & screen.inside(uv)


def four_step_phase(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    四步相移相位解算。

    条纹约定：
        I_k = a + b*cos(phi + [0, pi/2, pi, 3pi/2])
    """
    i0, i1, i2, i3 = images
    phase = np.arctan2(i3 - i1, i0 - i2)
    average = 0.25 * (i0 + i1 + i2 + i3)
    modulation = 0.5 * np.sqrt((i0 - i2) ** 2 + (i3 - i1) ** 2)
    return phase, average, modulation


def unwrap_phase_2d(phase: np.ndarray, direction: str) -> np.ndarray:
    """
    对包裹相位做二维展开。

    X 方向条纹先沿图像列方向展开，Y 方向条纹先沿图像行方向展开。
    这里仿真缺陷较小、无遮挡，numpy.unwrap 足够稳定；真实实验中可替换
    为质量引导展开、多频展开或 Gray Code 绝对相位。
    """
    if direction.lower() == "x":
        return np.unwrap(np.unwrap(phase, axis=1), axis=0)
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)


def align_unwrapped_phase_to_screen(
    unwrapped_phase: np.ndarray,
    screen_coord: np.ndarray,
    period_px: float,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """利用标定几何给解包裹相位补上全局 2pi 阶次/偏置。"""
    target = 2.0 * np.pi * screen_coord / period_px
    offset = float(np.median(target[mask] - unwrapped_phase[mask]))
    return unwrapped_phase + offset, offset


def remove_plane_trend(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """从相位图中拟合并去除 ax + by + c 平面趋势，便于观察局部缺陷扰动。"""
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.column_stack([xx[mask].ravel(), yy[mask].ravel(), np.ones(np.count_nonzero(mask))])
    b = data[mask].ravel()
    coeff, *_ = np.linalg.lstsq(a, b, rcond=None)
    plane = coeff[0] * xx + coeff[1] * yy + coeff[2]
    return data - plane


def estimate_slopes_from_phase_difference(
    camera: Camera,
    screen: Screen,
    reference_surface: GaussianMirror,
    reference_uv_x: np.ndarray,
    reference_uv_y: np.ndarray,
    dphi_x: np.ndarray,
    dphi_y: np.ndarray,
    period_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据相位差估计表面 X/Y 方向梯度。

    相位差对应屏幕坐标偏移：
        du = dphi_x * period / (2*pi)
        dv = dphi_y * period / (2*pi)

    然后使用参考平面交点 P 和偏移后的屏幕点 Q 估计被测面法向：
        n = normalize(normalize(C-P) + normalize(Q-P))

    对 z = h(x,y)，梯度满足：
        dz/dx = -nx/nz, dz/dy = -ny/nz
    """
    rays = camera.pixel_rays()
    points_ref = reference_surface.intersect(rays)

    du = dphi_x * period_px / (2.0 * np.pi)
    dv = dphi_y * period_px / (2.0 * np.pi)

    uv_obj = np.empty_like(reference_uv_x)
    uv_obj[..., 0] = reference_uv_x[..., 0] + du
    uv_obj[..., 1] = reference_uv_y[..., 1] + dv
    q_obj = screen.uv_to_world(uv_obj)

    vc = normalize(-points_ref)
    vs = normalize(q_obj - points_ref)
    n_obj = normalize(vc + vs)

    q_ref = screen.uv_to_world(np.stack([reference_uv_x[..., 0], reference_uv_y[..., 1]], axis=-1))
    vs_ref = normalize(q_ref - points_ref)
    n_ref = normalize(vc + vs_ref)

    p_obj = -n_obj[..., 0] / (n_obj[..., 2] + EPS)
    q_obj_slope = -n_obj[..., 1] / (n_obj[..., 2] + EPS)
    p_ref = -n_ref[..., 0] / (n_ref[..., 2] + EPS)
    q_ref_slope = -n_ref[..., 1] / (n_ref[..., 2] + EPS)
    return p_obj - p_ref, q_obj_slope - q_ref_slope, uv_obj


def integrate_poisson_dct(p: np.ndarray, q: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    用 DCT-Poisson 方法从梯度 p=dz/dx, q=dz/dy 积分恢复高度。

    高度的绝对零点不可由梯度确定，因此返回零均值相对高度。
    """
    dpdx = np.gradient(p, dx, axis=1, edge_order=2)
    dqdy = np.gradient(q, dy, axis=0, edge_order=2)
    div = dpdx + dqdy

    h, w = div.shape
    div_hat = dctn(div, type=2, norm="ortho")

    ky = np.arange(h)
    kx = np.arange(w)
    lambda_y = 2.0 * (np.cos(np.pi * ky / h) - 1.0) / (dy * dy)
    lambda_x = 2.0 * (np.cos(np.pi * kx / w) - 1.0) / (dx * dx)
    denom = lambda_y[:, None] + lambda_x[None, :]

    z_hat = np.zeros_like(div_hat)
    z_hat[denom != 0] = div_hat[denom != 0] / denom[denom != 0]
    z = idctn(z_hat, type=2, norm="ortho")
    z -= np.mean(z)
    return z


def align_zero_mean(z: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = z.copy()
    if mask is None:
        out -= np.mean(out)
    else:
        out -= np.mean(out[mask])
    return out


def save_scalar_image(
    path: Path,
    image: np.ndarray,
    title: str,
    cmap: str = "viridis",
    unit: str | None = None,
    mask: np.ndarray | None = None,
) -> None:
    data = np.array(image, dtype=np.float64)
    if mask is not None:
        data = np.where(mask, data, np.nan)
    plt.figure(figsize=(6.4, 4.8), dpi=130)
    im = plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    if unit:
        cb.set_label(unit)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_surface_3d(
    path: Path,
    z_data: np.ndarray,
    title: str,
    z_label: str,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    x_label: str = "X/pixel",
    y_label: str = "Y/pixel",
    cmap: str = "jet",
    mask: np.ndarray | None = None,
    stride: int | None = None,
    contour_projection: bool = False,
) -> None:
    """保存类似论文图中的 3D 相位/面型曲面图。"""
    data = np.asarray(z_data, dtype=np.float64)
    h, w = data.shape
    if x_grid is None or y_grid is None:
        yy, xx = np.mgrid[0:h, 0:w]
    else:
        xx = np.asarray(x_grid, dtype=np.float64)
        yy = np.asarray(y_grid, dtype=np.float64)

    if mask is not None:
        data = np.where(mask, data, np.nan)

    if stride is None:
        stride = max(1, min(h, w) // 150)
    xs = xx[::stride, ::stride]
    ys = yy[::stride, ::stride]
    zs = data[::stride, ::stride]
    zs_masked = np.ma.masked_invalid(zs)

    fig = plt.figure(figsize=(8.0, 5.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xs, ys, zs_masked, cmap=cmap, linewidth=0, antialiased=True, rcount=160, ccount=160)
    if contour_projection:
        finite = zs[np.isfinite(zs)]
        if finite.size:
            z_offset = float(np.nanmin(finite))
            ax.contourf(xs, ys, zs, zdir="z", offset=z_offset, levels=16, cmap=cmap, alpha=0.75)
            ax.set_zlim(z_offset, float(np.nanmax(finite)))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(elev=26, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """用兼容中文路径的方式保存 OpenCV 图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    buf.tofile(str(path))


def save_montage(path: Path, items: list[tuple[str, np.ndarray, str, str | None]]) -> None:
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    plt.figure(figsize=(4.8 * cols, 3.8 * rows), dpi=130)
    for idx, (title, image, cmap, unit) in enumerate(items, 1):
        ax = plt.subplot(rows, cols, idx)
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cb.set_label(unit)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_pattern_preview(path: Path, pattern: np.ndarray) -> None:
    preview = cv2.resize(pattern, (640, 360), interpolation=cv2.INTER_AREA)
    imwrite_unicode(path, preview)


def save_camera_steps(folder: Path, prefix: str, images: np.ndarray) -> None:
    """保存四步相移采集图。"""
    for idx, img in enumerate(images, 1):
        imwrite_unicode(folder / f"{prefix}_第{idx}步.png", img)


def save_geometry_plot(path: Path, camera: Camera, screen: Screen, reference: GaussianMirror, measured: GaussianMirror) -> None:
    """保存一个简化的 x-z 侧视几何示意图，便于检查系统参数。"""
    plt.figure(figsize=(7.2, 4.6), dpi=140)
    plt.scatter([0], [0], c="tab:red", label="相机中心")
    plt.plot([-screen.width_m / 2, screen.width_m / 2], [screen.z, screen.z], c="tab:blue", lw=3, label="LCD屏幕")
    xs = np.linspace(-0.45, 0.45, 300)
    ys = np.zeros_like(xs)
    plt.plot(xs, np.full_like(xs, reference.z0), c="tab:green", lw=2, label="参考平面")
    plt.plot(xs, measured.z0 + measured.height_xy(xs, ys), c="tab:orange", lw=2, label="被测面截面")
    plt.xlabel("x / m")
    plt.ylabel("z / m")
    plt.title("单目 PMD 仿真几何侧视图")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    output_dir = Path("results")
    if output_dir.exists():
        resolved = output_dir.resolve()
        if resolved.name != "results":
            raise RuntimeError(f"拒绝删除非 results 目录：{resolved}")
        shutil.rmtree(resolved)

    dirs = {
        "params": output_dir / "01_系统参数",
        "camera_calib": output_dir / "01_系统参数" / "01_相机标定过程",
        "reference_calib": output_dir / "01_系统参数" / "02_参考平面标定过程",
        "screen_calib": output_dir / "01_系统参数" / "03_屏幕标定过程",
        "screen": output_dir / "02_屏幕条纹图像",
        "ref": output_dir / "03_参考平面模拟采集图像",
        "obj": output_dir / "04_被测面模拟采集图像",
        "phase": output_dir / "05_相位解算结果",
        "phase_diff": output_dir / "06_相位差结果",
        "gradient": output_dir / "07_梯度结果",
        "recon": output_dir / "08_面型重建结果",
        "error": output_dir / "09_误差分析结果",
    }
    output_dir.mkdir(exist_ok=True)
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    # 真实系统参数只用于生成模拟采集图像；PMD 重建使用后续标定得到的参数。
    true_camera = Camera(width=480, height=360, fx=550.0, fy=550.0, cx=239.5, cy=179.5)
    true_screen = Screen(z=0.20, width_m=1.80, height_m=1.25, width_px=1920, height_px=1080)
    true_reference = GaussianMirror(z0=1.0, amplitude=0.0)
    measured = GaussianMirror(z0=1.0, amplitude=0.0015, sigma=0.09, x0=0.03, y0=-0.02)

    phase_steps = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    period_px = 96.0

    # 0）模拟真实标定过程：生成圆点板/屏幕圆点阵列采集图，再从图像中检测圆点并求参数。
    calibrated_camera, dist_est, camera_calib_rms, camera_calib_count = run_camera_calibration(
        true_camera, dirs["camera_calib"]
    )
    calibrated_reference, reference_reproj_error = run_reference_plane_calibration(
        true_camera, calibrated_camera, dist_est, true_reference, dirs["reference_calib"]
    )
    calibrated_screen, screen_reproj_error = run_screen_calibration(
        true_camera,
        calibrated_camera,
        dist_est,
        true_screen,
        true_reference,
        calibrated_reference,
        dirs["screen_calib"],
    )

    # 1）生成 LCD 屏幕上显示的 X/Y 方向四步相移正弦条纹。
    # 后续光线追迹会在这些连续条纹上采样，等价于真实相机拍摄屏幕反射像。
    lcd_x = make_screen_patterns(true_screen, "x", period_px, phase_steps)
    lcd_y = make_screen_patterns(true_screen, "y", period_px, phase_steps)
    for idx, pattern in enumerate(lcd_x, 1):
        save_pattern_preview(dirs["screen"] / f"X方向屏幕条纹_第{idx}步.png", pattern)
    for idx, pattern in enumerate(lcd_y, 1):
        save_pattern_preview(dirs["screen"] / f"Y方向屏幕条纹_第{idx}步.png", pattern)

    save_geometry_plot(dirs["params"] / "真实系统几何示意图.png", true_camera, true_screen, true_reference, measured)
    save_geometry_plot(dirs["params"] / "标定后系统几何示意图.png", calibrated_camera, calibrated_screen, calibrated_reference, measured)
    (dirs["params"] / "系统参数.txt").write_text(
        "\n".join(
            [
                "单目 PMD 全仿真系统参数",
                "",
                "【真实相机内参：仅用于生成模拟图像】",
                f"图像宽度 width = {true_camera.width} px",
                f"图像高度 height = {true_camera.height} px",
                f"fx = {true_camera.fx:.6f} px",
                f"fy = {true_camera.fy:.6f} px",
                f"cx = {true_camera.cx:.6f} px",
                f"cy = {true_camera.cy:.6f} px",
                "",
                "【标定得到的相机内参：用于后续 PMD 重建】",
                f"fx = {calibrated_camera.fx:.6f} px",
                f"fy = {calibrated_camera.fy:.6f} px",
                f"cx = {calibrated_camera.cx:.6f} px",
                f"cy = {calibrated_camera.cy:.6f} px",
                f"相机标定 RMS 重投影误差 = {camera_calib_rms:.6f} px",
                f"成功用于相机标定的图像数 = {camera_calib_count}",
                "",
                "【真实 LCD 屏幕：仅用于生成模拟图像】",
                f"屏幕平面 z = {true_screen.z:.6f} m",
                f"物理宽度 = {true_screen.width_m:.6f} m",
                f"物理高度 = {true_screen.height_m:.6f} m",
                f"像素宽度 = {true_screen.width_px} px",
                f"像素高度 = {true_screen.height_px} px",
                f"像素间距 pitch_x = {true_screen.pitch_x:.9f} m/px",
                f"像素间距 pitch_y = {true_screen.pitch_y:.9f} m/px",
                "",
                "【标定得到的参考平面与屏幕】",
                f"参考平面 z0 = {calibrated_reference.z0:.9f} m",
                f"参考平面 PnP 重投影误差 = {reference_reproj_error:.6f} px",
                f"屏幕平面 z = {calibrated_screen.z:.9f} m",
                f"屏幕虚像 PnP 重投影误差 = {screen_reproj_error:.6f} px",
                "",
                "【参考平面与被测面】",
                f"真实参考平面 z0 = {true_reference.z0:.6f} m",
                f"高斯缺陷幅值 A = {measured.amplitude:.9f} m",
                f"高斯缺陷 sigma = {measured.sigma:.9f} m",
                f"高斯缺陷中心 x0 = {measured.x0:.9f} m",
                f"高斯缺陷中心 y0 = {measured.y0:.9f} m",
                "",
                "【条纹】",
                f"四步相移 = {phase_steps}",
                f"条纹周期 = {period_px:.3f} LCD px",
            ]
        ),
        encoding="utf-8",
    )

    # 2）模拟参考平面和被测面的相机采集图像。
    # 这里没有直接使用真实高度求梯度，而是先生成相机图像，再从图像解相位。
    ref_x_imgs, ref_uv_x, ref_points, ref_valid_x = simulate_camera_images(
        true_camera, true_screen, true_reference, "x", period_px, phase_steps, noise_std=1.0, seed=1
    )
    ref_y_imgs, ref_uv_y, _, ref_valid_y = simulate_camera_images(
        true_camera, true_screen, true_reference, "y", period_px, phase_steps, noise_std=1.0, seed=2
    )
    obj_x_imgs, obj_uv_x_true, obj_points, obj_valid_x = simulate_camera_images(
        true_camera, true_screen, measured, "x", period_px, phase_steps, noise_std=1.0, seed=3
    )
    obj_y_imgs, obj_uv_y_true, _, obj_valid_y = simulate_camera_images(
        true_camera, true_screen, measured, "y", period_px, phase_steps, noise_std=1.0, seed=4
    )
    ref_uv_est, ref_points_est, ref_geom_valid = compute_reference_uv_from_geometry(
        calibrated_camera, calibrated_screen, calibrated_reference
    )
    valid = ref_valid_x & ref_valid_y & obj_valid_x & obj_valid_y & ref_geom_valid

    save_camera_steps(dirs["ref"], "X方向参考条纹", ref_x_imgs)
    save_camera_steps(dirs["ref"], "Y方向参考条纹", ref_y_imgs)
    save_camera_steps(dirs["obj"], "X方向被测条纹", obj_x_imgs)
    save_camera_steps(dirs["obj"], "Y方向被测条纹", obj_y_imgs)

    # 3）四步相移相位解算，得到参考平面和被测面的包裹相位。
    phi_ref_x, _, mod_ref_x = four_step_phase(ref_x_imgs)
    phi_ref_y, _, mod_ref_y = four_step_phase(ref_y_imgs)
    phi_obj_x, _, mod_obj_x = four_step_phase(obj_x_imgs)
    phi_obj_y, _, mod_obj_y = four_step_phase(obj_y_imgs)

    # 4）解包裹相位，并用标定几何补全全局相位阶次，得到 absolute phase/rad。
    phi_ref_x_unwrap_raw = unwrap_phase_2d(phi_ref_x, "x")
    phi_ref_y_unwrap_raw = unwrap_phase_2d(phi_ref_y, "y")
    phi_obj_x_unwrap_raw = unwrap_phase_2d(phi_obj_x, "x")
    phi_obj_y_unwrap_raw = unwrap_phase_2d(phi_obj_y, "y")
    phi_ref_x_abs, phase_offset_x = align_unwrapped_phase_to_screen(
        phi_ref_x_unwrap_raw, ref_uv_est[..., 0], period_px, valid
    )
    phi_ref_y_abs, phase_offset_y = align_unwrapped_phase_to_screen(
        phi_ref_y_unwrap_raw, ref_uv_est[..., 1], period_px, valid
    )
    phi_obj_x_abs = phi_obj_x_unwrap_raw + phase_offset_x
    phi_obj_y_abs = phi_obj_y_unwrap_raw + phase_offset_y
    phi_obj_x_abs_detrend = remove_plane_trend(phi_obj_x_abs, valid)
    phi_obj_y_abs_detrend = remove_plane_trend(phi_obj_y_abs, valid)

    # 5）计算参考面与被测面的相位差。差分 PMD 主要使用这个观测量。
    dphi_x = wrap_to_pi(phi_obj_x - phi_ref_x)
    dphi_y = wrap_to_pi(phi_obj_y - phi_ref_y)
    dphi_x_abs = phi_obj_x_abs - phi_ref_x_abs
    dphi_y_abs = phi_obj_y_abs - phi_ref_y_abs

    # 6）把相位差换算为屏幕坐标偏移，再利用反射定律估计 X/Y 梯度。
    grad_x, grad_y, obj_uv_est = estimate_slopes_from_phase_difference(
        calibrated_camera,
        calibrated_screen,
        calibrated_reference,
        ref_uv_est,
        ref_uv_est,
        dphi_x_abs,
        dphi_y_abs,
        period_px,
    )
    grad_x = np.where(valid, grad_x, 0.0)
    grad_y = np.where(valid, grad_y, 0.0)

    # 7）在参考平面坐标网格上对梯度进行 DCT-Poisson 积分。
    dx = calibrated_reference.z0 / calibrated_camera.fx
    dy = calibrated_reference.z0 / calibrated_camera.fy
    z_rec = integrate_poisson_dct(grad_x, grad_y, dx, dy)

    # 8）只在误差评估阶段采样真实高度。注意：真实高度没有参与相位和梯度重建。
    rays = calibrated_camera.pixel_rays()
    p_ref = calibrated_reference.intersect(rays)
    z_true = measured.height_xy(p_ref[..., 0], p_ref[..., 1])
    z_true = align_zero_mean(z_true, valid)
    z_rec = align_zero_mean(z_rec, valid)
    z_err = np.where(valid, z_rec - z_true, np.nan)

    rms = float(np.sqrt(np.nanmean(z_err[valid] ** 2)))
    pv = float(np.nanmax(z_err[valid]) - np.nanmin(z_err[valid]))
    true_pv = float(np.nanmax(z_true[valid]) - np.nanmin(z_true[valid]))
    rec_pv = float(np.nanmax(z_rec[valid]) - np.nanmin(z_rec[valid]))
    mae = float(np.nanmean(np.abs(z_err[valid])))
    max_error = float(np.nanmax(np.abs(z_err[valid])))
    x_mm = p_ref[..., 0] * 1e3
    y_mm = p_ref[..., 1] * 1e3

    # 9）保存相位、相位差、梯度、面型和误差分析图片。
    save_scalar_image(dirs["phase"] / "X方向参考包裹相位.png", phi_ref_x, "X方向参考包裹相位", "twilight", "rad", valid)
    save_scalar_image(dirs["phase"] / "Y方向参考包裹相位.png", phi_ref_y, "Y方向参考包裹相位", "twilight", "rad", valid)
    save_scalar_image(dirs["phase"] / "X方向被测包裹相位.png", phi_obj_x, "X方向被测包裹相位", "twilight", "rad", valid)
    save_scalar_image(dirs["phase"] / "Y方向被测包裹相位.png", phi_obj_y, "Y方向被测包裹相位", "twilight", "rad", valid)
    save_surface_3d(dirs["phase"] / "X方向参考包裹相位_3D.png", phi_ref_x, "X方向参考包裹相位", "phase/rad", cmap="jet", mask=valid)
    save_surface_3d(dirs["phase"] / "Y方向参考包裹相位_3D.png", phi_ref_y, "Y方向参考包裹相位", "phase/rad", cmap="jet", mask=valid)
    save_surface_3d(dirs["phase"] / "X方向被测包裹相位_3D.png", phi_obj_x, "X方向被测包裹相位", "phase/rad", cmap="jet", mask=valid)
    save_surface_3d(dirs["phase"] / "Y方向被测包裹相位_3D.png", phi_obj_y, "Y方向被测包裹相位", "phase/rad", cmap="jet", mask=valid)
    save_scalar_image(dirs["phase"] / "X方向参考解包裹绝对相位.png", phi_ref_x_abs, "X方向参考解包裹绝对相位", "jet", "rad", valid)
    save_scalar_image(dirs["phase"] / "Y方向参考解包裹绝对相位.png", phi_ref_y_abs, "Y方向参考解包裹绝对相位", "jet", "rad", valid)
    save_scalar_image(dirs["phase"] / "X方向被测解包裹绝对相位.png", phi_obj_x_abs, "X方向被测解包裹绝对相位", "jet", "rad", valid)
    save_scalar_image(dirs["phase"] / "Y方向被测解包裹绝对相位.png", phi_obj_y_abs, "Y方向被测解包裹绝对相位", "jet", "rad", valid)
    save_scalar_image(dirs["phase"] / "X方向被测解包裹绝对相位_去平面趋势.png", phi_obj_x_abs_detrend, "X方向被测解包裹绝对相位 去平面趋势", "coolwarm", "rad", valid)
    save_scalar_image(dirs["phase"] / "Y方向被测解包裹绝对相位_去平面趋势.png", phi_obj_y_abs_detrend, "Y方向被测解包裹绝对相位 去平面趋势", "coolwarm", "rad", valid)
    save_surface_3d(dirs["phase"] / "X方向参考解包裹绝对相位_3D.png", phi_ref_x_abs, "X方向参考解包裹绝对相位", "Absolute phase/rad", cmap="jet", mask=valid, contour_projection=True)
    save_surface_3d(dirs["phase"] / "Y方向参考解包裹绝对相位_3D.png", phi_ref_y_abs, "Y方向参考解包裹绝对相位", "Absolute phase/rad", cmap="jet", mask=valid, contour_projection=True)
    save_surface_3d(dirs["phase"] / "X方向被测解包裹绝对相位_3D.png", phi_obj_x_abs, "X方向被测解包裹绝对相位", "Absolute phase/rad", cmap="jet", mask=valid, contour_projection=True)
    save_surface_3d(dirs["phase"] / "Y方向被测解包裹绝对相位_3D.png", phi_obj_y_abs, "Y方向被测解包裹绝对相位", "Absolute phase/rad", cmap="jet", mask=valid, contour_projection=True)
    save_surface_3d(dirs["phase"] / "X方向被测解包裹绝对相位_去平面趋势_3D.png", phi_obj_x_abs_detrend, "X方向被测解包裹绝对相位 去平面趋势", "phase residual/rad", cmap="coolwarm", mask=valid, contour_projection=True)
    save_surface_3d(dirs["phase"] / "Y方向被测解包裹绝对相位_去平面趋势_3D.png", phi_obj_y_abs_detrend, "Y方向被测解包裹绝对相位 去平面趋势", "phase residual/rad", cmap="coolwarm", mask=valid, contour_projection=True)
    save_scalar_image(dirs["phase"] / "X方向调制度图.png", mod_ref_x, "X方向参考调制度", "gray", None, valid)
    save_scalar_image(dirs["phase"] / "Y方向调制度图.png", mod_ref_y, "Y方向参考调制度", "gray", None, valid)

    save_scalar_image(dirs["phase_diff"] / "X方向相位差.png", dphi_x, "X方向包裹相位差", "coolwarm", "rad", valid)
    save_scalar_image(dirs["phase_diff"] / "Y方向相位差.png", dphi_y, "Y方向包裹相位差", "coolwarm", "rad", valid)
    save_scalar_image(dirs["phase_diff"] / "X方向解包裹相位差.png", dphi_x_abs, "X方向解包裹相位差", "coolwarm", "rad", valid)
    save_scalar_image(dirs["phase_diff"] / "Y方向解包裹相位差.png", dphi_y_abs, "Y方向解包裹相位差", "coolwarm", "rad", valid)
    save_surface_3d(dirs["phase_diff"] / "X方向相位差_3D.png", dphi_x_abs, "X方向解包裹相位差", "phase/rad", cmap="coolwarm", mask=valid)
    save_surface_3d(dirs["phase_diff"] / "Y方向相位差_3D.png", dphi_y_abs, "Y方向解包裹相位差", "phase/rad", cmap="coolwarm", mask=valid)

    save_scalar_image(dirs["gradient"] / "X方向梯度图.png", grad_x, "X方向梯度 dz/dx", "coolwarm", None, valid)
    save_scalar_image(dirs["gradient"] / "Y方向梯度图.png", grad_y, "Y方向梯度 dz/dy", "coolwarm", None, valid)
    save_surface_3d(dirs["gradient"] / "X方向梯度图_3D.png", grad_x, "X方向梯度 dz/dx", "dz/dx", cmap="coolwarm", mask=valid)
    save_surface_3d(dirs["gradient"] / "Y方向梯度图_3D.png", grad_y, "Y方向梯度 dz/dy", "dz/dy", cmap="coolwarm", mask=valid)

    save_scalar_image(dirs["recon"] / "真实高度图.png", z_true * 1e6, "真实高度图", "viridis", "um", valid)
    save_scalar_image(dirs["recon"] / "重建高度图.png", z_rec * 1e6, "重建高度图", "viridis", "um", valid)
    save_scalar_image(dirs["recon"] / "高度误差图.png", z_err * 1e6, "高度误差图", "coolwarm", "um", valid)
    save_surface_3d(dirs["recon"] / "真实高度三维面型图.png", z_true * 1e3, "真实高度三维面型图", "Z/mm", x_grid=x_mm, y_grid=y_mm, x_label="X/mm", y_label="Y/mm", cmap="jet", mask=valid, contour_projection=True)
    save_surface_3d(dirs["recon"] / "重建高度三维面型图.png", z_rec * 1e3, "重建高度三维面型图", "Z/mm", x_grid=x_mm, y_grid=y_mm, x_label="X/mm", y_label="Y/mm", cmap="jet", mask=valid, contour_projection=True)
    save_surface_3d(dirs["recon"] / "高度误差三维图.png", z_err * 1e6, "高度误差三维图", "error/um", x_grid=x_mm, y_grid=y_mm, x_label="X/mm", y_label="Y/mm", cmap="coolwarm", mask=valid, contour_projection=True)

    save_scalar_image(dirs["error"] / "高度绝对误差图.png", np.abs(z_err) * 1e6, "高度绝对误差图", "magma", "um", valid)

    save_montage(
        dirs["error"] / "结果总览图.png",
        [
            ("参考 X 条纹", ref_x_imgs[0], "gray", None),
            ("被测 X 条纹", obj_x_imgs[0], "gray", None),
            ("被测 X 包裹相位", phi_obj_x, "twilight", "rad"),
            ("X 方向相位差", dphi_x_abs, "coolwarm", "rad"),
            ("Y 方向相位差", dphi_y_abs, "coolwarm", "rad"),
            ("X 方向梯度", grad_x, "coolwarm", None),
            ("Y 方向梯度", grad_y, "coolwarm", None),
            ("重建高度 / um", z_rec * 1e6, "viridis", "um"),
            ("高度误差 / um", z_err * 1e6, "coolwarm", "um"),
        ],
    )

    # 10）保存 summary.txt，便于复现实验参数和误差指标。
    report = output_dir / "summary.txt"
    report.write_text(
        "\n".join(
            [
                "单目相位偏折术 PMD 全仿真 summary",
                "",
                "【主要参数】",
                f"高斯缺陷幅值 A = {measured.amplitude:.9e} m = {measured.amplitude * 1e6:.3f} um",
                f"高斯缺陷 sigma = {measured.sigma:.9e} m",
                f"高斯缺陷中心 x0 = {measured.x0:.9e} m",
                f"高斯缺陷中心 y0 = {measured.y0:.9e} m",
                f"真实参考平面 z0 = {true_reference.z0:.9e} m",
                f"标定参考平面 z0 = {calibrated_reference.z0:.9e} m",
                f"相机分辨率 = {true_camera.width} x {true_camera.height}",
                f"真实相机内参 fx, fy = {true_camera.fx:.3f}, {true_camera.fy:.3f} px",
                f"标定相机内参 fx, fy = {calibrated_camera.fx:.3f}, {calibrated_camera.fy:.3f} px",
                f"相机标定 RMS = {camera_calib_rms:.6f} px",
                f"参考平面 PnP 重投影误差 = {reference_reproj_error:.6f} px",
                f"屏幕 PnP 重投影误差 = {screen_reproj_error:.6f} px",
                f"屏幕分辨率 = {true_screen.width_px} x {true_screen.height_px}",
                f"真实屏幕平面 z = {true_screen.z:.9e} m",
                f"标定屏幕平面 z = {calibrated_screen.z:.9e} m",
                f"条纹周期 = {period_px:.3f} LCD px",
                f"有效像素数 = {int(np.count_nonzero(valid))} / {valid.size}",
                f"X方向绝对相位偏置 = {phase_offset_x:.9f} rad",
                f"Y方向绝对相位偏置 = {phase_offset_y:.9f} rad",
                "",
                "【误差指标】",
                f"真实高度 PV = {true_pv * 1e6:.6f} um",
                f"重建高度 PV = {rec_pv * 1e6:.6f} um",
                f"重建高度 RMSE = {rms * 1e6:.6f} um",
                f"重建高度 MAE = {mae * 1e6:.6f} um",
                f"最大误差 = {max_error * 1e6:.6f} um",
                f"高度误差 PV = {pv * 1e6:.6f} um",
            ]
        ),
        encoding="utf-8",
    )

    print(f"结果已保存到：{output_dir.resolve()}")
    print(f"有效像素：{int(np.count_nonzero(valid))} / {valid.size}")
    print(f"真实高度 PV：{true_pv * 1e6:.3f} um")
    print(f"重建高度 PV：{rec_pv * 1e6:.3f} um")
    print(f"重建高度 RMSE：{rms * 1e6:.3f} um")
    print(f"重建高度 MAE：{mae * 1e6:.3f} um")
    print(f"最大误差：{max_error * 1e6:.3f} um")


if __name__ == "__main__":
    main()
