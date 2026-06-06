"""
PMD pattern display and camera capture controller.

功能：
1. 按顺序全屏显示 LCD 图案；
2. 等待屏幕稳定；
3. 触发相机拍照；
4. 保存采集图像，文件名与投射图案序号一一对应。

推荐硬件：
    Daheng MER2-502-79U3C + Galaxy SDK Python(gxipy)

也提供 OpenCV VideoCapture 模式，方便先调通显示/保存流程。

示例：
    # 采参考平面镜
    python pmd_capture_sequence.py --camera gx --output captures/reference

    # 采被测镜
    python pmd_capture_sequence.py --camera gx --output captures/object

    # 如果 LCD 是扩展屏，且在主屏右侧，从 x=1920 开始显示
    python pmd_capture_sequence.py --camera gx --screen-x 1920 --screen-y 0 --output captures/reference

    # 没有 Galaxy SDK 时，用 OpenCV 摄像头先测试流程
    python pmd_capture_sequence.py --camera opencv --opencv-index 0 --output captures/test
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """兼容中文路径保存图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(str(path))


def read_sequence(pattern_root: Path) -> list[tuple[int, str, Path]]:
    """
    优先读取采集顺序.csv。
    若没有 csv，则读取 07_完整采集序列 下的 png 文件。
    """
    csv_path = pattern_root / "采集顺序.csv"
    seq_dir = pattern_root / "07_完整采集序列"
    sequence: list[tuple[int, str, Path]] = []

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["序号"])
                purpose = row["用途"]
                filename = row["文件名"]
                numbered = seq_dir / f"{idx:03d}_{filename}"
                plain = seq_dir / filename
                path = numbered if numbered.exists() else plain
                if not path.exists():
                    raise FileNotFoundError(f"找不到序列图片: {path}")
                sequence.append((idx, purpose, path))
    else:
        images = sorted(seq_dir.glob("*.png"))
        for i, path in enumerate(images, 1):
            sequence.append((i, "pattern", path))

    if not sequence:
        raise RuntimeError(f"没有找到可显示的序列图片: {pattern_root}")
    return sequence


class OpenCVDisplay:
    def __init__(self, window_name: str, screen_x: int, screen_y: int, fullscreen: bool) -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, screen_x, screen_y)
        if fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, image_path: Path) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"读取图案失败: {image_path}")
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
        return img

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)


class OpenCVCamera:
    def __init__(self, index: int, width: int | None, height: int | None) -> None:
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开相机 index={index}")

    def capture(self) -> np.ndarray:
        # 丢几帧，避免拿到旧图像。
        frame = None
        for _ in range(3):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("OpenCV 相机采集失败")
        assert frame is not None
        return frame

    def close(self) -> None:
        self.cap.release()


class GxCamera:
    """
    Daheng Galaxy SDK gxipy 相机封装。

    需要先安装 Galaxy SDK，并确认 Python 能 import gxipy。
    """

    def __init__(
        self,
        exposure_us: float | None,
        gain: float | None,
        trigger: bool,
        timeout_ms: int,
    ) -> None:
        try:
            import gxipy as gx
        except ImportError as exc:
            raise RuntimeError("未找到 gxipy。请先安装大恒 Galaxy SDK Python 接口。") from exc

        self.gx = gx
        self.timeout_ms = timeout_ms
        manager = gx.DeviceManager()
        dev_num, _ = manager.update_device_list()
        if dev_num < 1:
            raise RuntimeError("没有发现大恒相机，请检查 USB3.0 连接和 Galaxy SDK。")

        self.cam = manager.open_device_by_index(1)

        if exposure_us is not None and hasattr(self.cam, "ExposureTime"):
            self.cam.ExposureTime.set(float(exposure_us))
        if gain is not None and hasattr(self.cam, "Gain"):
            self.cam.Gain.set(float(gain))

        self.trigger = trigger
        if self.trigger:
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        else:
            self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        self.cam.stream_on()

    def capture(self) -> np.ndarray:
        if self.trigger:
            self.cam.TriggerSoftware.send_command()

        raw = self.cam.data_stream[0].get_image(self.timeout_ms)
        if raw is None:
            raise RuntimeError("gxipy 获取图像超时")

        # 彩色相机一般需要转 RGB；若当前像素格式已是单通道，则直接保存灰度。
        try:
            rgb = raw.convert("RGB")
            arr = rgb.get_numpy_array()
            if arr is None:
                raise RuntimeError("gxipy RGB 图像为空")
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            arr = raw.get_numpy_array()
            if arr is None:
                raise RuntimeError("gxipy 图像为空")
            return arr

    def close(self) -> None:
        self.cam.stream_off()
        self.cam.close_device()


def wait_key_or_continue(delay_ms: int, allow_abort: bool = True) -> None:
    start = time.time()
    while (time.time() - start) * 1000 < delay_ms:
        key = cv2.waitKey(10) & 0xFF
        if allow_abort and key in (27, ord("q"), ord("Q")):
            raise KeyboardInterrupt("用户终止采集")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PMD LCD 图案显示与相机采集控制")
    parser.add_argument(
        "--pattern-root",
        type=Path,
        default=Path("PMD实验预备图片_1920x1080"),
        help="图案根目录，默认使用前一步生成的 PMD实验预备图片_1920x1080",
    )
    parser.add_argument("--output", type=Path, default=Path("captures/reference"), help="采集图像保存目录")
    parser.add_argument("--camera", choices=["gx", "opencv"], default="gx", help="相机接口")
    parser.add_argument("--opencv-index", type=int, default=0, help="OpenCV 相机索引")
    parser.add_argument("--opencv-width", type=int, default=None, help="OpenCV 模式设置宽度")
    parser.add_argument("--opencv-height", type=int, default=None, help="OpenCV 模式设置高度")
    parser.add_argument("--exposure-us", type=float, default=None, help="gxipy 曝光时间，单位 us；不填则不修改")
    parser.add_argument("--gain", type=float, default=None, help="gxipy 增益；不填则不修改")
    parser.add_argument("--no-trigger", action="store_true", help="gxipy 使用连续采集而不是软件触发")
    parser.add_argument("--timeout-ms", type=int, default=3000, help="相机取图超时")
    parser.add_argument("--settle-ms", type=int, default=250, help="每张图显示后等待屏幕稳定的时间")
    parser.add_argument("--screen-x", type=int, default=0, help="显示窗口左上角 x 坐标，多屏时可设为第二屏起点")
    parser.add_argument("--screen-y", type=int, default=0, help="显示窗口左上角 y 坐标")
    parser.add_argument("--windowed", action="store_true", help="不用全屏窗口，调试用")
    parser.add_argument("--manual", action="store_true", help="每张图等待按空格再拍照")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = read_sequence(args.pattern_root)
    args.output.mkdir(parents=True, exist_ok=True)

    display = OpenCVDisplay("PMD_LCD_PATTERN", args.screen_x, args.screen_y, fullscreen=not args.windowed)
    if args.camera == "gx":
        camera = GxCamera(args.exposure_us, args.gain, trigger=not args.no_trigger, timeout_ms=args.timeout_ms)
    else:
        camera = OpenCVCamera(args.opencv_index, args.opencv_width, args.opencv_height)

    log_path = args.output / "capture_log.csv"
    try:
        with log_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "用途", "投射图案", "采集图像", "时间戳"])

            for idx, purpose, pattern_path in sequence:
                display.show(pattern_path)

                if args.manual:
                    print(f"[{idx:03d}] 显示 {pattern_path.name}，按空格拍照，按 q/ESC 退出")
                    while True:
                        key = cv2.waitKey(20) & 0xFF
                        if key == ord(" "):
                            break
                        if key in (27, ord("q"), ord("Q")):
                            raise KeyboardInterrupt("用户终止采集")
                else:
                    wait_key_or_continue(args.settle_ms)

                frame = camera.capture()
                out_name = f"{idx:03d}_{purpose}_{pattern_path.stem}.png"
                out_path = args.output / out_name
                imwrite_unicode(out_path, frame)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([idx, purpose, pattern_path.name, out_path.name, timestamp])
                print(f"[{idx:03d}/{len(sequence):03d}] saved: {out_path.name}")

    except KeyboardInterrupt as exc:
        print(f"采集已终止: {exc}")
    finally:
        camera.close()
        display.close()

    print(f"采集完成，输出目录: {args.output.resolve()}")


if __name__ == "__main__":
    main()
