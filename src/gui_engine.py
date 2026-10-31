"""Bộ xử lý ảnh/video/camera cho cửa sổ GUI.

Tất cả callback được gọi trong luồng nền. GUI KHÔNG cập nhật widget trực tiếp
trong callback; giao tiếp qua queue trên luồng Tk chính.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
import time

import cv2
import numpy as np

from .common import (
    RunLogger, RunStats, annotate_frame, extract_detections, get_images,
    load_yolo, predict_frame,
)


@dataclass(frozen=True)
class GuiConfig:
    mode: str
    model: str
    source: str = ""
    confidence: float = 0.32
    iou: float = 0.60
    imgsz: int = 640
    camera_id: int = 0
    record_camera: bool = False
    output_root: str = "outputs"


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _write_image_unicode(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Không mã hóa được ảnh: {path}")
    buffer.tofile(str(path))


def _read_image_unicode(path: Path):
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def _run_images(cfg, model, stopped: Event, frame_cb, message_cb):
    images = get_images(Path(cfg.source))
    output_dir = Path(cfg.output_root) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = RunStats("image", cfg.source, output=str(output_dir))
    log = RunLogger(str(Path(cfg.output_root) / "logs"), "image")
    started = time.perf_counter()
    try:
        for ix, image_path in enumerate(images, 1):
            if stopped.is_set():
                break
            frame = _read_image_unicode(image_path)
            if frame is None:
                message_cb(f"Bỏ qua ảnh không đọc được: {image_path.name}")
                continue
            tic = time.perf_counter()
            result = predict_frame(model, frame, conf=cfg.confidence,
                                   iou=cfg.iou, imgsz=cfg.imgsz)
            counts, detections = extract_detections(result)
            annotated = annotate_frame(frame, result, counts)
            target = output_dir / f"{image_path.stem}_result.jpg"
            _write_image_unicode(target, annotated)
            stats.add(counts)
            log.write(str(image_path), stats.frames, detections)
            fps = 1.0 / max(time.perf_counter() - tic, 1e-6)
            frame_cb(annotated, counts, fps, ix, len(images), image_path.name)
            message_cb(f"Đã lưu: {target.name}")
    finally:
        stats.elapsed_seconds = time.perf_counter() - started
        log.close(stats)
    return stats, log.csv_path


def _run_stream(cfg, model, stopped: Event, frame_cb, message_cb):
    camera = cfg.mode == "camera"
    source = cfg.camera_id if camera else cfg.source
    if not camera and not Path(cfg.source).is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {cfg.source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Không mở được {'webcam ID ' + str(source) if camera else 'video ' + str(source)}."
        )

    output = None
    if cfg.mode == "video" or cfg.record_camera:
        subdir = "camera" if camera else "videos"
        output = Path(cfg.output_root) / subdir / f"{cfg.mode}_result_{_timestamp()}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

    stats = RunStats(cfg.mode, f"camera:{cfg.camera_id}" if camera else cfg.source,
                     output=str(output) if output else "Chỉ xem trực tiếp (không ghi video)")
    logger = RunLogger(str(Path(cfg.output_root) / "logs"), cfg.mode)
    fps_input = float(cap.get(cv2.CAP_PROP_FPS))
    if not 5 <= fps_input <= 120:
        fps_input = 25.0
    total = 0 if camera else max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    writer = None
    started = time.perf_counter()
    smooth_fps = None
    last_preview = 0.0
    try:
        while not stopped.is_set():
            tic = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if camera:
                    message_cb("Camera ngừng cung cấp khung hình.")
                break
            result = predict_frame(model, frame, conf=cfg.confidence,
                                   iou=cfg.iou, imgsz=cfg.imgsz)
            counts, detections = extract_detections(result)
            stats.add(counts)
            logger.write(stats.source, stats.frames, detections)
            instant_fps = 1.0 / max(time.perf_counter() - tic, 1e-6)
            smooth_fps = instant_fps if smooth_fps is None else smooth_fps * 0.8 + instant_fps * 0.2
            annotated = annotate_frame(frame, result, counts, smooth_fps)
            if output is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_input, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Không tạo được video MP4 đầu ra (codec mp4v).")
                writer.write(annotated)
            now = time.perf_counter()
            if now - last_preview >= 1 / 20 or (total and stats.frames == total):
                frame_cb(annotated, counts, smooth_fps, stats.frames, total,
                         f"Camera {cfg.camera_id}" if camera else Path(cfg.source).name)
                last_preview = now
        if stopped.is_set():
            message_cb("Đã dừng theo yêu cầu.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        stats.elapsed_seconds = time.perf_counter() - started
        logger.close(stats)
    return stats, logger.csv_path


def process(cfg: GuiConfig, stop_event: Event, frame_cb, message_cb):
    """Chạy đồng bộ trong thread. Trả về (RunStats, đường dẫn log CSV)."""
    if cfg.mode not in {"image", "video", "camera"}:
        raise ValueError("Chế độ không hợp lệ.")
    if not 0 < cfg.confidence <= 1:
        raise ValueError("Confidence phải trong khoảng (0, 1].")
    if not Path(cfg.model).is_file():
        raise FileNotFoundError(f"Không có model: {cfg.model}. Hãy đặt best_v1.pt vào models/.")
    if cfg.mode != "camera" and not cfg.source:
        raise ValueError("Bạn chưa chọn ảnh/video đầu vào.")
    message_cb("Đang tải mô hình YOLO11s...")
    model = load_yolo(cfg.model)
    message_cb("Đã tải model. Đang nhận diện...")
    if cfg.mode == "image":
        return _run_images(cfg, model, stop_event, frame_cb, message_cb)
    return _run_stream(cfg, model, stop_event, frame_cb, message_cb)
