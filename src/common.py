"""Cấu hình và công cụ dùng chung cho ảnh, video và camera."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time

DEFAULT_MODEL = "models/best_v1.pt"
DEFAULT_CONF = 0.32
DEFAULT_IOU = 0.60
DEFAULT_IMGSZ = 640
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = ["timestamp", "mode", "source", "frame_index", "class_name", "confidence", "x1", "y1", "x2", "y2"]


def category_from_name(name: str) -> str:
    """Phân biệt hai lớp dựa vào TÊN class (không đoán số class ID)."""
    compact = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    if compact in {"nohelmet", "withouthelmet", "notwearinghelmet", "nonhelmet"}:
        return "no_helmet"
    if compact in {"helmet", "withhelmet", "wearinghelmet"}:
        return "helmet"
    return "other"


def get_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Định dạng ảnh không được hỗ trợ: {input_path}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"Thư mục không có ảnh hỗ trợ: {input_path}")
        return images
    raise FileNotFoundError(f"Không tìm thấy ảnh/thư mục đầu vào: {input_path}")


def load_yolo(model_path: str):
    """Tải YOLO duy nhất một lần cho mỗi lượt chạy (không tải trong vòng lặp frame)."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có trọng số mô hình: {path}\n"
            "Hãy chép file best_v1.pt đã huấn luyện vào models/ "
            "hoặc cung cấp đường dẫn bằng --model."
        )
    from ultralytics import YOLO
    return YOLO(str(path))


def predict_frame(model, frame, *, conf: float, iou: float, imgsz: int, device: str | None = None):
    arguments = dict(source=frame, conf=conf, iou=iou, imgsz=imgsz,
                     agnostic_nms=False, verbose=False)
    if device:
        arguments["device"] = device
    results = model.predict(**arguments)
    return results[0] if results else None


def extract_detections(result) -> tuple[dict, list[dict]]:
    """Đọc bounding box, class và confidence theo định dạng Ultralytics."""
    counts = {"helmet": 0, "no_helmet": 0, "other": 0}
    rows = []
    if result is None or result.boxes is None:
        return counts, rows
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        name = str(result.names[class_id])
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (int(round(x)) for x in box.xyxy[0].tolist())
        counts[category_from_name(name)] += 1
        rows.append({"class_name": name, "confidence": round(confidence, 4),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return counts, rows


def annotate_frame(frame, result, counts: dict, fps: float | None = None):
    import cv2
    output = result.plot() if result is not None else frame.copy()
    lines = [f"Helmet: {counts['helmet']} | No Helmet: {counts['no_helmet']}"]
    if counts["other"]:
        lines.append(f"Other class: {counts['other']}")
    if fps is not None:
        lines.append(f"Processing FPS: {fps:.1f}")
    width = output.shape[1]
    for idx, line in enumerate(lines):
        y = 29 + idx * 29
        rect_right = min(width - 1, max(1, len(line) * 11 + 17))
        cv2.rectangle(output, (0, y - 24), (rect_right, y + 6), (25, 25, 25), -1)
        cv2.putText(output, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return output


@dataclass
class RunStats:
    mode: str
    source: str
    frames: int = 0
    frames_with_detection: int = 0
    helmet: int = 0
    no_helmet: int = 0
    other: int = 0
    elapsed_seconds: float = 0.0
    output: str = ""

    def add(self, counts: dict):
        self.frames += 1
        if sum(counts.values()):
            self.frames_with_detection += 1
        self.helmet += counts["helmet"]
        self.no_helmet += counts["no_helmet"]
        self.other += counts["other"]


class RunLogger:
    def __init__(self, log_dir: str, mode: str):
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.csv_path = directory / f"{mode}_detections_{suffix}.csv"
        self.summary_path = directory / f"{mode}_summary_{suffix}.txt"
        self._handle = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.mode = mode

    def write(self, source: str, index: int, detections: list[dict]):
        time_value = datetime.now().isoformat(timespec="seconds")
        for detection in detections:
            self.writer.writerow({"timestamp": time_value, "mode": self.mode,
                                  "source": source, "frame_index": index, **detection})

    def close(self, stats: RunStats):
        if not self._handle.closed:
            self._handle.close()
        self.summary_path.write_text(
            f"Thời gian: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Chế độ: {stats.mode}\nNguồn: {stats.source}\n"
            f"Số ảnh/frame đã xử lý: {stats.frames}\n"
            f"Số ảnh/frame có phát hiện: {stats.frames_with_detection}\n"
            f"Tổng lượt phát hiện helmet: {stats.helmet}\n"
            f"Tổng lượt phát hiện no_helmet: {stats.no_helmet}\n"
            f"Tổng lượt phát hiện class khác: {stats.other}\n"
            f"Thời gian xử lý (giây): {stats.elapsed_seconds:.2f}\n"
            f"Output: {stats.output}\nLog chi tiết: {self.csv_path}\n"
            "Chú ý: lượt phát hiện trên video/camera là tổng qua các frame, "
            "KHÔNG phải số người duy nhất.\n",
            encoding="utf-8",
        )


def display_results(stats: RunStats, logger: RunLogger):
    print("\n" + "=" * 56)
    print(f"HOÀN TẤT DEMO: {stats.mode.upper()}")
    print(f"Đã xử lý: {stats.frames} ảnh/frame; có phát hiện: {stats.frames_with_detection}")
    print(f"Helmet: {stats.helmet}; No Helmet: {stats.no_helmet}; Khác: {stats.other}")
    print(f"Thời gian: {stats.elapsed_seconds:.2f} giây")
    print(f"Kết quả: {stats.output}")
    print(f"Log CSV: {logger.csv_path}")
    print(f"Báo cáo: {logger.summary_path}")
    print("Lưu ý: tổng lượt detection của video/camera có thể đếm lặp người qua các frame.")
    print("=" * 56)


def maybe_show(window_name: str, image, wait_ms: int) -> bool:
    """Trả về False khi người dùng nhấn q hoặc ESC."""
    import cv2
    try:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(wait_ms) & 0xFF
    except cv2.error as exc:
        raise RuntimeError("Không mở được cửa sổ OpenCV. Hãy bỏ --show hoặc chạy trên máy có màn hình.") from exc
    return key not in (27, ord("q"))


def now_seconds():
    return time.perf_counter()
