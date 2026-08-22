# -*- coding: utf-8 -*-
import argparse
import json
import locale
import mimetypes
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid
import webbrowser


import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np



def configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass



configure_stdio()


SCRIPT_ROOT = Path(__file__).resolve().parent

WORKFLOW_SCRIPT = SCRIPT_ROOT / "host_train_export.py"
TEST_SCRIPT = SCRIPT_ROOT / "model_test.py"
LABEL_SCRIPT = SCRIPT_ROOT / "video_track_label.py"
USER_DEFAULTS_FILE = SCRIPT_ROOT / "train_panel_defaults.json"
STOP_EXPORT_SIGNAL_FILE = SCRIPT_ROOT / ".train_stop_export.signal"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}

TRAIN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}






DEFAULT_VALUES: dict[str, Any] = {
    "dataset_root": str(SCRIPT_ROOT),
    "train_task": "detect",
    "train_images_dir": str(SCRIPT_ROOT / "images"),
    "train_annotations_dir": str(SCRIPT_ROOT / "annotations"),
    "train_ratio_percent": "80",
    "img_width": "448",
    "img_height": "448",
    "image_resize_mode": "letterbox",

    "epochs": "200",
    "batch": "16",
    "lr0": "0.005",
    "conda_env": "yolov8",
    "base_model": "yolov8n.pt",
    "torch_cuda": "cu128",
    "train_device": "cuda",
    "train_cache": "False",

    "project_name": "douzi_yolov8n_448",
    "model_name": "douzi_yolov8n_448",
    "test_model": "",
    "test_source": "camera",
    "test_image_file": "",
    "test_image_folder": "",
    "test_output_dir": "",
    "camera_index": "0",
    "conf": "0.25",
    "label_video_dir": "",
    "label_video": "",
    "label_camera_index": "0",
    "label_source_type": "video",
    "label_images_input_dir": "",
    "label_name": "w",

    "label_interval": "5",
    "label_images_dir": str(SCRIPT_ROOT / "images"),
    "label_annotations_dir": str(SCRIPT_ROOT / "annotations"),
    "label_prefix": "track",
    "label_tracker": "csrt",
    "label_start_frame": "0",
    "label_max_frames": "0",
    "label_display_scale": "1.0",
    "label_jpeg_quality": "95",
}





STATE_LOCK = threading.RLock()
STATE: dict[str, Any] = {
    "values": DEFAULT_VALUES.copy(),
    "logs": [],
    "markers": {},
    "train_progress": {
        "phase": "idle",
        "task": "",
        "epoch": 0,
        "total_epochs": 0,
        "batch": 0,
        "total_batches": 0,
        "percent": 0.0,
        "gpu_mem": "",
        "loss": None,
        "box_loss": None,
        "cls_loss": None,
        "dfl_loss": None,
        "instances": None,
        "size": None,
        "speed": "",
        "elapsed": "",
        "eta": "",
        "val_batch": 0,
        "val_total": 0,
        "val_percent": 0.0,
        "metrics": {},
        "history": [],
        "updated_at": "",
    },
    "running": False,
    "job": None,
    "exit_code": None,
    "started_at": None,
    "finished_at": None,
    "last_error": "",
}
@dataclass
class LabelTrackObject:
    obj_id: int
    label: str
    bbox: tuple[int, int, int, int]
    tracker: object
    ok: bool = True
    sample_count: int = 1
    hidden: bool = False


class TemplateTracker:
    def __init__(self, search_scale: float = 2.5, min_score: float = 0.45):
        self.search_scale = search_scale
        self.min_score = min_score
        self.template = None
        self.bbox: Optional[tuple[int, int, int, int]] = None

    def init(self, frame, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 2 or h <= 2:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y + h, x:x + w].copy()
        self.bbox = (x, y, w, h)
        return True

    def update(self, frame):
        if self.template is None or self.bbox is None:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        pad_x, pad_y = max(int(w * self.search_scale), 20), max(int(h * self.search_scale), 20)
        sx1, sy1 = max(0, x - pad_x), max(0, y - pad_y)
        sx2, sy2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
        search = gray[sy1:sy2, sx1:sx2]
        if search.shape[0] < h or search.shape[1] < w:
            return False, self.bbox
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        self.bbox = sanitize_label_bbox((sx1 + loc[0], sy1 + loc[1], w, h), frame.shape[1], frame.shape[0])
        return score >= self.min_score, self.bbox


class MultiTemplateTracker:
    """以 CSRT 为主跟踪，多个参考模板仅用于 CSRT 丢失后的恢复。"""
    def __init__(self, search_scale: float = 2.5, min_score: float = 0.55):
        self.search_scale = search_scale
        self.min_score = min_score
        self.samples: list[tuple[Any, int, int]] = []
        self.bbox: Optional[tuple[int, int, int, int]] = None
        self.primary = None

    def _reset_primary(self, frame, bbox: tuple[int, int, int, int]) -> None:
        self.primary = make_label_cv_tracker("csrt")
        if self.primary is not None and not init_label_tracker(self.primary, frame, bbox):
            self.primary = None

    def init(self, frame, bbox: tuple[int, int, int, int]) -> bool:
        if make_label_cv_tracker("csrt") is None:
            raise RuntimeError("Multi-template 需要 OpenCV CSRT；请安装 opencv-contrib-python 后重启标注工具。")
        self.samples = []
        self.bbox = None
        self.primary = None
        return self.add_sample(frame, bbox)

    def add_sample(self, frame, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 2 or h <= 2:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.samples.append((gray[y:y + h, x:x + w].copy(), w, h))
        self.bbox = (x, y, w, h)
        self._reset_primary(frame, self.bbox)
        return True

    def _recover_from_templates(self, frame):
        if self.bbox is None:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        pad_x, pad_y = max(int(w * self.search_scale), 20), max(int(h * self.search_scale), 20)
        sx1, sy1 = max(0, x - pad_x), max(0, y - pad_y)
        sx2, sy2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
        search = gray[sy1:sy2, sx1:sx2]
        best_score, best_bbox = -1.0, self.bbox
        for template, sample_w, sample_h in self.samples:
            if search.shape[0] < sample_h or search.shape[1] < sample_w:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_bbox = sanitize_label_bbox((sx1 + loc[0], sy1 + loc[1], sample_w, sample_h), frame.shape[1], frame.shape[0])
        if best_score < self.min_score:
            return False, self.bbox
        self.bbox = best_bbox
        self._reset_primary(frame, self.bbox)
        return True, self.bbox

    def update(self, frame):
        if not self.samples or self.bbox is None:
            return False, (0, 0, 0, 0)
        if self.primary is not None:
            ok, bbox = self.primary.update(frame)
            if ok:
                self.bbox = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
                return True, self.bbox
        return self._recover_from_templates(frame)


def sanitize_label_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [int(round(value)) for value in bbox]
    x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


def make_label_cv_tracker(name: str):
    upper = name.upper()
    candidates = []
    if hasattr(cv2, "legacy"):
        candidates.append((cv2.legacy, f"Tracker{upper}_create"))
    candidates.append((cv2, f"Tracker{upper}_create"))
    for module, factory in candidates:
        if hasattr(module, factory):
            return getattr(module, factory)()
    return None


def make_label_tracker(name: str):
    normalized = name.lower()
    if normalized == "template":
        return TemplateTracker()
    if normalized == "multi_template":
        return MultiTemplateTracker()
    tracker = make_label_cv_tracker(name)
    if tracker is not None:
        return tracker
    append_log(f"[网页标注] 跟踪器 {name} 不可用，已回退到 Template。\n")
    return TemplateTracker()


def init_label_tracker(tracker, frame, bbox: tuple[int, int, int, int]) -> bool:
    result = tracker.init(frame, bbox)
    return True if result is None else bool(result)


def label_object_data(obj: LabelTrackObject) -> dict[str, Any]:
    x, y, w, h = obj.bbox
    return {
        "id": obj.obj_id, "label": obj.label, "x": x, "y": y, "w": w, "h": h,
        "ok": obj.ok, "sample_count": obj.sample_count, "hidden": obj.hidden,
    }


LABEL_SESSIONS: dict[str, dict[str, Any]] = {}
LABEL_SESSIONS_LOCK = threading.RLock()


MAX_LOG_LINES = 3000
MAX_LABEL_VIDEOS = 2000
MAX_VIDEO_PREVIEW_WIDTH = 560
VIDEO_PREVIEW_CACHE_LIMIT = 80
VIDEO_PREVIEW_CACHE: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
VIDEO_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
}





def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def clean_values(values: Optional[dict[str, Any]]) -> dict[str, Any]:

    merged = DEFAULT_VALUES.copy()
    if values:
        source = values.copy()
        legacy_img_size = source.get("img_size")
        if legacy_img_size is not None:
            source.setdefault("img_width", legacy_img_size)
            source.setdefault("img_height", legacy_img_size)
        for key in merged:
            if key in source:
                merged[key] = as_bool(source[key]) if isinstance(DEFAULT_VALUES[key], bool) else str(source[key])
    return merged


def load_user_defaults() -> dict[str, Any]:
    if not USER_DEFAULTS_FILE.is_file():
        return DEFAULT_VALUES.copy()
    try:
        data = json.loads(USER_DEFAULTS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"无法读取默认配置 {USER_DEFAULTS_FILE}: {exc}", file=sys.stderr)
        return DEFAULT_VALUES.copy()
    return clean_values(data if isinstance(data, dict) else None)


def save_user_defaults(values: dict[str, Any]) -> dict[str, Any]:
    defaults = clean_values(values)
    tmp_path = USER_DEFAULTS_FILE.with_suffix(USER_DEFAULTS_FILE.suffix + ".tmp")
    tmp_path.write_text(json.dumps(defaults, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(USER_DEFAULTS_FILE)
    return defaults


def quote_cmd(cmd: list[Any]) -> str:
    return shlex.join([str(x) for x in cmd])


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def subprocess_creationflags() -> int:
    return 0


def terminate_process_tree(proc: subprocess.Popen[Any], timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()



def decode_process_output(raw: bytes) -> str:
    encodings = ["utf-8", locale.getpreferredencoding(False), "gb18030"]
    tried: set[str] = set()

    for encoding in encodings:
        normalized = (encoding or "").lower()
        if not normalized or normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def append_log(text: str) -> None:
    with STATE_LOCK:
        STATE["logs"].append(text)
        if len(STATE["logs"]) > MAX_LOG_LINES:
            STATE["logs"] = STATE["logs"][-MAX_LOG_LINES:]


def strip_ansi(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return text.replace("\r", "").strip()


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_gib(value: float) -> str:
    if value <= 0:
        return "0 GB"
    return f"{value:.1f} GB" if value < 100 else f"{value:.0f} GB"


MODEL_RESOURCE_PROFILES: dict[str, dict[str, float]] = {
    "n": {"base_vram": 0.80, "per_image_vram": 0.025, "base_ram": 1.70},
    "s": {"base_vram": 1.10, "per_image_vram": 0.045, "base_ram": 1.95},
    "m": {"base_vram": 1.60, "per_image_vram": 0.080, "base_ram": 2.35},
    "l": {"base_vram": 2.20, "per_image_vram": 0.130, "base_ram": 2.90},
    "x": {"base_vram": 3.00, "per_image_vram": 0.190, "base_ram": 3.60},
}
MAX_IMAGE_SIZE_SAMPLES = 300
LOCAL_RESOURCE_CACHE: dict[str, Any] = {"updated_at": 0.0, "data": None}


def infer_model_size(base_model: str) -> str:

    name = Path(str(base_model or "")).name.lower()
    match = re.search(r"yolo(?:v?\d+)?([nslmx])(?:[._-]|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[._-])([nslmx])(?:[._-]|$)", name)
    return match.group(1) if match else "n"


def read_image_size(path: Path) -> Optional[tuple[int, int]]:
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                return struct.unpack(">II", head[16:24])
            if head.startswith(b"BM") and len(head) >= 26:
                return struct.unpack("<II", head[18:26])
            if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                if head[12:16] == b"VP8X" and len(head) >= 30:
                    return (int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1)
                if head[12:16] == b"VP8L" and len(head) >= 25:
                    b0, b1, b2, b3 = head[21:25]
                    width = 1 + (((b1 & 0x3F) << 8) | b0)
                    height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                    return width, height
            if head.startswith(b"\xff\xd8"):
                fh.seek(2)
                while True:
                    marker = fh.read(2)
                    if len(marker) < 2:
                        return None
                    while marker[0] != 0xFF:
                        marker = marker[1:] + fh.read(1)
                        if len(marker) < 2:
                            return None
                    code = marker[1]
                    if code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                        data = fh.read(7)
                        if len(data) < 7:
                            return None
                        return struct.unpack(">HH", data[3:7])[::-1]
                    if code in {0xD8, 0xD9, 0x01} or 0xD0 <= code <= 0xD7:
                        continue
                    size_data = fh.read(2)
                    if len(size_data) < 2:
                        return None
                    segment_size = struct.unpack(">H", size_data)[0]
                    if segment_size < 2:
                        return None
                    fh.seek(segment_size - 2, 1)
    except OSError:
        return None
    return None


def empty_local_resources() -> dict[str, Any]:
    return {"ram_total_gib": None, "ram_available_gib": None, "gpu_name": "", "gpu_total_gib": None, "gpu_free_gib": None}


def query_local_resources(cache_seconds: float = 5.0) -> dict[str, Any]:
    now = time.time()
    cached = LOCAL_RESOURCE_CACHE.get("data")
    if isinstance(cached, dict) and now - float(LOCAL_RESOURCE_CACHE.get("updated_at") or 0.0) < cache_seconds:
        return cached.copy()

    resources = empty_local_resources()
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        resources["ram_total_gib"] = mem.total / (1024 ** 3)
        resources["ram_available_gib"] = mem.available / (1024 ** 3)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        lines = (result.stdout or "").strip().splitlines()
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                resources["gpu_name"] = parts[0]
                resources["gpu_total_gib"] = float(parts[1]) / 1024
                resources["gpu_free_gib"] = float(parts[2]) / 1024
    except Exception:
        pass
    LOCAL_RESOURCE_CACHE["updated_at"] = now
    LOCAL_RESOURCE_CACHE["data"] = resources.copy()
    return resources



def image_dimensions(values: dict[str, Any]) -> tuple[int, int]:
    width = max(32, parse_int(str(values.get("img_width", ""))) or 448)
    height = max(32, parse_int(str(values.get("img_height", ""))) or 448)
    return width, height


def estimate_train_resources(values: dict[str, Any]) -> dict[str, Any]:
    img_width, img_height = image_dimensions(values)
    img_pixels = img_width * img_height
    batch = max(1, parse_int(str(values.get("batch", ""))) or 1)
    cache_mode = str(values.get("train_cache") or "False")
    train_device = str(values.get("train_device") or "cuda")
    base_model = str(values.get("base_model") or "")

    train_task = str(values.get("train_task") or "detect")
    model_size = infer_model_size(base_model)
    profile = MODEL_RESOURCE_PROFILES[model_size]
    if train_task == "classify":
        profile = {
            "base_vram": profile["base_vram"] * 0.65,
            "per_image_vram": profile["per_image_vram"] * 0.55,
            "base_ram": profile["base_ram"] * 0.8,
        }
    image_count = 0
    image_bytes = 0
    sampled_images = 0
    sampled_pixels = 0
    images_dir = Path(str(values.get("train_images_dir") or "")).expanduser()
    if images_dir.is_dir():
        try:
            for path in images_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS:
                    image_count += 1
                    try:
                        image_bytes += path.stat().st_size
                    except OSError:
                        pass
                    if sampled_images < MAX_IMAGE_SIZE_SAMPLES:
                        size = read_image_size(path)
                        if size:
                            width, height = size
                            if width > 0 and height > 0:
                                sampled_images += 1
                                sampled_pixels += width * height
        except OSError:
            pass

    img_scale = img_pixels / (640 * 640)
    avg_pixels = sampled_pixels / sampled_images if sampled_images else img_pixels
    raw_cache_gib = image_count * avg_pixels * 3 / (1024 ** 3) if image_count else 0.0
    disk_cache_gib = raw_cache_gib * 1.20
    cache_ram_gib = raw_cache_gib * 1.15 if cache_mode == "True" else 0.0
    cache_disk_gib = disk_cache_gib if cache_mode == "disk" else 0.0

    batch_tensor_gib = batch * img_pixels * 3 * 4 / (1024 ** 3)
    loader_ram_gib = min(4.0, batch_tensor_gib * 2.2 + 0.35)
    augment_ram_gib = min(3.0, batch_tensor_gib * 1.4 + 0.25)
    safety_ram_gib = max(0.6, (profile["base_ram"] + loader_ram_gib + augment_ram_gib + cache_ram_gib) * 0.15)
    train_ram_gib = profile["base_ram"] + loader_ram_gib + augment_ram_gib + safety_ram_gib
    total_ram_gib = train_ram_gib + cache_ram_gib

    raw_vram_gib = profile["base_vram"] + batch * profile["per_image_vram"] * img_scale
    train_vram_gib = raw_vram_gib * 1.18 + 0.25 if train_device == "cuda" else 0.0

    local = query_local_resources()
    risk = "safe"
    risk_text = "资源预估"
    if train_device == "cuda" and local.get("gpu_free_gib") is not None:

        free_vram = float(local["gpu_free_gib"])
        if train_vram_gib > free_vram * 0.95:
            risk, risk_text = "danger", "显存可能不足"
        elif train_vram_gib > free_vram * 0.75:
            risk, risk_text = "warning", "显存接近上限"
    if local.get("ram_available_gib") is not None:

        free_ram = float(local["ram_available_gib"])
        if total_ram_gib > free_ram * 0.95:
            risk, risk_text = "danger", "内存可能不足"
        elif risk == "safe" and total_ram_gib > free_ram * 0.75:
            risk, risk_text = "warning", "内存接近上限"

    notes = [
        f"模型档位={model_size}，显存按 base {profile['base_vram']:.1f}GB + batch*{profile['per_image_vram']:.3f}GB*(imgsz/640)^2 估算。",
        f"RAM=训练 {format_gib(train_ram_gib)} + cache {format_gib(cache_ram_gib)}。",
    ]
    if sampled_images:
        notes.append(f"cache 按真实图片尺寸采样 {sampled_images}/{image_count} 张估算。")
    elif not image_count:
        notes.append("未读取到图片数量，cache 部分按 0 估算。")
    else:
        notes.append("未能读取图片尺寸，cache 暂按 ImgSize 估算。")
    if cache_mode == "True":
        notes.append("内存 cache 会额外占用 RAM。")
    elif cache_mode == "disk":
        notes.append("disk cache 主要额外占用磁盘。")
    if train_device != "cuda":
        notes.append("当前选择 CPU 训练，显存按 0 估算。")
    elif local.get("gpu_free_gib") is not None:
        notes.append(f"当前 GPU 可用约 {format_gib(float(local['gpu_free_gib']))}。")


    return {
        "image_count": image_count,
        "image_bytes": image_bytes,
        "sampled_images": sampled_images,
        "avg_image_mp": round(avg_pixels / 1_000_000, 2) if image_count else 0,
        "img_size": f"{img_width} × {img_height}",
        "batch": batch,
        "base_model": base_model,
        "model_size": model_size,
        "cache_mode": cache_mode,

        "risk": risk,
        "risk_text": risk_text,
        "ram_gib": round(total_ram_gib, 2),
        "vram_gib": round(train_vram_gib, 2),
        "train_ram_gib": round(train_ram_gib, 2),
        "cache_ram_gib": round(cache_ram_gib, 2),
        "cache_disk_gib": round(cache_disk_gib, 2),
        "ram_text": format_gib(total_ram_gib),
        "vram_text": "0 GB" if train_device != "cuda" else format_gib(train_vram_gib),
        "cache_text": format_gib(cache_ram_gib) if cache_mode == "True" else (format_gib(cache_disk_gib) if cache_mode == "disk" else "0 GB"),
        "local_resources": local,
        "note": " ".join(notes),
    }



def reset_train_progress_locked() -> None:

    STATE["train_progress"] = {
        "phase": "idle",
        "task": "",
        "epoch": 0,
        "total_epochs": 0,
        "batch": 0,
        "total_batches": 0,
        "percent": 0.0,
        "gpu_mem": "",
        "loss": None,
        "box_loss": None,
        "cls_loss": None,
        "dfl_loss": None,
        "instances": None,
        "size": None,
        "speed": "",
        "elapsed": "",
        "eta": "",
        "val_batch": 0,
        "val_total": 0,
        "val_percent": 0.0,
        "metrics": {},
        "history": [],
        "updated_at": "",
    }


def append_epoch_history(progress: dict[str, Any]) -> None:
    epoch = progress.get("epoch")
    if not epoch:
        return
    history = progress.setdefault("history", [])
    item = {
        "epoch": epoch,
        "loss": progress.get("loss"),
        "box_loss": progress.get("box_loss"),
        "cls_loss": progress.get("cls_loss"),
        "dfl_loss": progress.get("dfl_loss"),
        "precision": progress.get("metrics", {}).get("precision"),
        "recall": progress.get("metrics", {}).get("recall"),
        "map50": progress.get("metrics", {}).get("map50"),
        "map50_95": progress.get("metrics", {}).get("map50_95"),
        "top1_acc": progress.get("metrics", {}).get("top1_acc"),
        "top5_acc": progress.get("metrics", {}).get("top5_acc"),
    }
    if history and history[-1].get("epoch") == epoch:
        history[-1] = item
    else:
        history.append(item)
    if len(history) > 500:
        del history[:-500]


def parse_train_output(line: str) -> None:
    clean = strip_ansi(line)
    if not clean:
        return
    now = time.strftime("%H:%M:%S")
    detect_match = re.search(
        r"(?P<epoch>\d+)\s*/\s*(?P<total_epochs>\d+)\s+(?P<gpu_mem>\S+)\s+"
        r"(?P<box_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"(?P<cls_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"(?P<dfl_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"(?P<instances>\d+)\s+(?P<size>\d+)\s*:\s*(?P<percent>\d+(?:\.\d+)?)%.*?"
        r"(?P<batch>\d+)\s*/\s*(?P<total_batches>\d+)"
        r"(?:\s+(?P<speed>[\d.]+it/s))?(?:\s+(?P<elapsed>[^<\s]+))?(?:<(?P<eta>\S+))?",
        clean,
    )
    if detect_match:
        data = detect_match.groupdict()
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({
                "phase": "train", "task": "detect", "epoch": parse_int(data["epoch"]) or 0,
                "total_epochs": parse_int(data["total_epochs"]) or 0, "batch": parse_int(data["batch"]) or 0,
                "total_batches": parse_int(data["total_batches"]) or 0, "percent": parse_float(data["percent"]) or 0.0,
                "gpu_mem": data.get("gpu_mem") or "", "loss": None, "box_loss": parse_float(data["box_loss"]),
                "cls_loss": parse_float(data["cls_loss"]), "dfl_loss": parse_float(data["dfl_loss"]),
                "instances": parse_int(data["instances"]), "size": parse_int(data["size"]),
                "speed": data.get("speed") or "", "elapsed": data.get("elapsed") or "", "eta": data.get("eta") or "",
                "updated_at": now,
            })
            if progress["percent"] >= 100:
                append_epoch_history(progress)
        return

    classify_match = re.search(
        r"(?P<epoch>\d+)\s*/\s*(?P<total_epochs>\d+)\s+(?P<gpu_mem>\S+)\s+"
        r"(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<instances>\d+)\s+(?P<size>\d+)\s*:\s*"
        r"(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)\s*/\s*(?P<total_batches>\d+)"
        r"(?:\s+(?P<speed>[\d.]+it/s))?(?:\s+(?P<elapsed>[^<\s]+))?(?:<(?P<eta>\S+))?",
        clean,
    )
    if classify_match:
        data = classify_match.groupdict()
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({
                "phase": "train", "task": "classify", "epoch": parse_int(data["epoch"]) or 0,
                "total_epochs": parse_int(data["total_epochs"]) or 0, "batch": parse_int(data["batch"]) or 0,
                "total_batches": parse_int(data["total_batches"]) or 0, "percent": parse_float(data["percent"]) or 0.0,
                "gpu_mem": data.get("gpu_mem") or "", "loss": parse_float(data["loss"]), "box_loss": None,
                "cls_loss": None, "dfl_loss": None, "instances": parse_int(data["instances"]),
                "size": parse_int(data["size"]), "speed": data.get("speed") or "", "elapsed": data.get("elapsed") or "",
                "eta": data.get("eta") or "", "updated_at": now,
            })
            if progress["percent"] >= 100:
                append_epoch_history(progress)
        return

    val_match = re.search(r"Class\s+Images\s+Instances.*?:\s*(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)\s*/\s*(?P<total>\d+)", clean)
    if val_match:
        data = val_match.groupdict()
        with STATE_LOCK:
            STATE["train_progress"].update({"phase": "val", "task": "detect", "val_batch": parse_int(data["batch"]) or 0, "val_total": parse_int(data["total"]) or 0, "val_percent": parse_float(data["percent"]) or 0.0, "updated_at": now})
        return

    classify_val_match = re.search(r"classes\s+top1_acc\s+top5_acc\s*:\s*(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)\s*/\s*(?P<total>\d+)", clean, re.IGNORECASE)
    if classify_val_match:
        data = classify_val_match.groupdict()
        with STATE_LOCK:
            STATE["train_progress"].update({"phase": "val", "task": "classify", "val_batch": parse_int(data["batch"]) or 0, "val_total": parse_int(data["total"]) or 0, "val_percent": parse_float(data["percent"]) or 0.0, "updated_at": now})
        return

    detect_metric_match = re.match(r"^all\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+(?P<precision>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<recall>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<map50>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<map50_95>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", clean)
    if detect_metric_match:
        data = detect_metric_match.groupdict()
        metrics = {"images": parse_int(data["images"]), "instances": parse_int(data["instances"]), "precision": parse_float(data["precision"]), "recall": parse_float(data["recall"]), "map50": parse_float(data["map50"]), "map50_95": parse_float(data["map50_95"])}
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({"phase": "metrics", "task": "detect", "metrics": metrics, "val_percent": 100.0, "updated_at": now})
            append_epoch_history(progress)
        return

    classify_metric_match = re.match(r"^all\s+(?P<top1_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(?P<top5_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", clean, re.IGNORECASE)
    if classify_metric_match:
        data = classify_metric_match.groupdict()
        metrics = {"top1_acc": parse_float(data["top1_acc"]), "top5_acc": parse_float(data["top5_acc"])}
        with STATE_LOCK:
            progress = STATE["train_progress"]
            progress.update({"phase": "metrics", "task": "classify", "metrics": metrics, "val_percent": 100.0, "updated_at": now})
            append_epoch_history(progress)



def parse_marker(line: str) -> None:
    markers = {
        "TRAIN_MODEL_ONNX=": "train_model_onnx",
        "TRAIN_PLOT_DIR=": "train_plot_dir",
        "TRAIN_MODEL_PT=": "test_model",
        "TRAIN_STOP_EXPORT_REQUESTED=": "stop_export",
        "TEST_OUTPUT_IMAGE=": "test_output_image",

    }
    with STATE_LOCK:
        for prefix, key in markers.items():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                STATE["markers"][key] = value
                if key in STATE["values"]:
                    STATE["values"][key] = value
                return


def build_common_args(values: dict[str, Any], stage: str) -> list[Any]:
    return [
        sys.executable,
        str(WORKFLOW_SCRIPT),
        "--stage", stage,
        "--dataset-root", values["dataset_root"],
        "--train-task", values["train_task"],
        "--images-dir", values["train_images_dir"],
        "--annotations-dir", values["train_annotations_dir"],
        "--train-ratio-percent", values["train_ratio_percent"],
        "--img-width", values["img_width"],
        "--img-height", values["img_height"],
        "--image-resize-mode", values["image_resize_mode"],

        "--epochs", values["epochs"],
        "--batch", values["batch"],
        "--lr0", values["lr0"],
        "--conda-env", values["conda_env"],
        "--base-model", values["base_model"],
        "--torch-cuda", values["torch_cuda"],
        "--train-device", values["train_device"],
        "--train-cache", values["train_cache"],
        "--stop-export-signal", str(STOP_EXPORT_SIGNAL_FILE),
        "--project-name", values["project_name"],


        "--model-name", values["model_name"],
    ]


def build_train_cmd(values: dict[str, Any]) -> list[Any]:
    return build_common_args(values, "train")


def build_test_cmd(values: dict[str, Any]) -> list[Any]:
    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--model", values["test_model"],
        "--source", values["test_source"],
        "--camera-index", values["camera_index"],
        "--conf", values["conf"],
    ]
    if values["test_source"] == "image":
        cmd += ["--image", values["test_image_file"]]
    elif values["test_source"] == "folder":
        cmd += ["--folder", values["test_image_folder"]]
        if values["test_output_dir"].strip():
            cmd += ["--output-dir", values["test_output_dir"]]
    return cmd


def build_label_cmd(values: dict[str, Any]) -> list[Any]:
    source_type = values.get("label_source_type", "video")
    cmd = [
        sys.executable,
        str(LABEL_SCRIPT),
        "--labels", values["label_name"],
        "--interval", values["label_interval"],
        "--images-dir", values["label_images_dir"],
        "--annotations-dir", values["label_annotations_dir"],
        "--prefix", values["label_prefix"],
        "--tracker", values["label_tracker"],
        "--start-frame", values["label_start_frame"],
        "--max-frames", values["label_max_frames"],
        "--display-scale", values["label_display_scale"],
        "--jpeg-quality", values["label_jpeg_quality"],
    ]
    if source_type == "images":
        cmd += ["--images-input-dir", values["label_images_input_dir"]]
    elif source_type == "camera":
        cmd += ["--video", values["label_camera_index"]]
    else:
        cmd += ["--video", values["label_video"]]
    return cmd



def command_for(action: str, values: dict[str, Any]) -> list[Any]:

    if action == "train":
        return build_train_cmd(values)
    if action == "test":
        return build_test_cmd(values)
    if action == "label":
        return build_label_cmd(values)
    raise ValueError(f"unknown action: {action}")


def validate(action: str, values: dict[str, Any]) -> None:
    if action == "train":
        dataset = Path(values["dataset_root"])
        images_dir = Path(values["train_images_dir"])
        annotations_dir = Path(values["train_annotations_dir"])
        if not dataset.is_dir():
            raise ValueError("Dataset Root 必须是有效文件夹，用于保存训练输出。")
        if not images_dir.is_dir():
            raise ValueError("Images Dir 必须是有效图片文件夹。")
        train_task = values.get("train_task", "detect")
        if train_task not in {"detect", "classify"}:
            raise ValueError("训练任务必须为目标检测或图像分类。")
        if train_task == "detect" and not annotations_dir.is_dir():
            raise ValueError("目标检测需要有效的 XML 标注文件夹。")
        if train_task == "classify":
            class_dirs = [path for path in images_dir.iterdir() if path.is_dir()]
            if len(class_dirs) < 2:
                raise ValueError("图像分类的 Images Dir 下至少需要两个类别子文件夹。")
        train_ratio = parse_float(str(values.get("train_ratio_percent", "")))
        if train_ratio is None or not 1 <= train_ratio <= 100:
            raise ValueError("训练集比例必须在 1% 到 100% 之间。")
        for key, label in (("img_width", "图片宽度"), ("img_height", "图片高度")):
            value = parse_int(str(values.get(key, "")))
            if value is None or value < 32 or value % 32:
                raise ValueError(f"{label}必须是大于等于 32 的 32 倍数。")
        if values.get("image_resize_mode") not in {"crop", "letterbox", "stretch"}:
            raise ValueError("图片适配方式必须为裁剪、等比缩放或拉伸。")
        if not WORKFLOW_SCRIPT.exists():
            raise ValueError(f"未找到 host_train_export.py: {WORKFLOW_SCRIPT}")
    elif action == "test":
        if not TEST_SCRIPT.exists():
            raise ValueError(f"未找到 model_test.py: {TEST_SCRIPT}")
        if not values["test_model"].strip():
            raise ValueError("测试模型不能为空。")
        if values["test_source"] == "image" and not values["test_image_file"].strip():
            raise ValueError("选择单张图片测试时必须填写图片路径。")
        if values["test_source"] == "folder" and not values["test_image_folder"].strip():
            raise ValueError("选择图片文件夹测试时必须填写文件夹路径。")
    elif action == "label":
        if not LABEL_SCRIPT.exists():
            raise ValueError(f"未找到 video_track_label.py: {LABEL_SCRIPT}")
        source_type = values.get("label_source_type", "video")
        if source_type == "images":
            image_dir = Path(values.get("label_images_input_dir", "").strip())
            if not image_dir.is_dir():
                raise ValueError("图片集文件夹必须是有效文件夹。")
        elif source_type == "camera":
            camera_index = values.get("label_camera_index", "").strip()
            if not camera_index.isdigit():
                raise ValueError("摄像头索引必须是非负整数，例如 0 或 1。")
        elif not values["label_video"].strip():
            raise ValueError("请先从视频队列选择或填写视频路径。")
        if not values["label_name"].strip():
            raise ValueError("请先填写至少一个标签名称。")


def start_job(action: str, values: dict[str, Any]) -> None:
    global current_proc, stop_requested

    validate(action, values)
    cmd = command_for(action, values)
    with STATE_LOCK:
        if STATE["running"]:
            raise RuntimeError("已有任务正在运行，请等待完成或先停止。")
        current_proc = None
        stop_requested = False
        STATE["values"] = values.copy()
        STATE["logs"] = []
        STATE["running"] = True
        STATE["job"] = action
        STATE["exit_code"] = None
        STATE["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE["finished_at"] = None
        STATE["last_error"] = ""
        if action == "train":
            reset_train_progress_locked()
            STATE["markers"].pop("stop_export", None)
            try:
                STOP_EXPORT_SIGNAL_FILE.unlink(missing_ok=True)
            except OSError:
                pass
            STATE["train_progress"]["phase"] = "pending"
            STATE["train_progress"]["updated_at"] = time.strftime("%H:%M:%S")



    def worker() -> None:
        global current_proc, stop_requested

        proc: Optional[subprocess.Popen[Any]] = None
        try:
            process_env = subprocess_env()
            process_cmd = list(cmd)
            append_log("$ " + quote_cmd(cmd) + "\n")
            proc = subprocess.Popen(
                [str(x) for x in process_cmd],
                cwd=str(SCRIPT_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=process_env,
                creationflags=subprocess_creationflags(),
                start_new_session=True,
            )
            with STATE_LOCK:
                current_proc = proc
                should_stop = stop_requested
            if should_stop:
                terminate_process_tree(proc)

            if proc.stdout is not None:
                buffer = bytearray()
                while True:
                    chunk = proc.stdout.read(1)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    preview = decode_process_output(bytes(buffer))
                    is_boundary = chunk in (b"\n", b"\r")
                    if is_boundary or len(buffer) >= 256:
                        line = preview
                        buffer.clear()
                        append_log(line)
                        if is_boundary:
                            parse_marker(line.strip())
                            if action == "train":
                                parse_train_output(line)
                if buffer:
                    line = decode_process_output(bytes(buffer))
                    append_log(line)
                    parse_marker(line.strip())
                    if action == "train":
                        parse_train_output(line)

            code = proc.wait()

            with STATE_LOCK:
                stopped = stop_requested
                STATE["exit_code"] = code
            if stopped:
                append_log(f"\n[stopped, exit code {code}]\n")
            else:
                append_log(f"\n[exit code {code}]\n")
        except Exception as exc:
            with STATE_LOCK:
                stopped = stop_requested
                STATE["exit_code"] = -15 if stopped else -1
                STATE["last_error"] = "" if stopped else str(exc)
            if stopped:
                append_log("\n[stopped]\n")
            else:
                append_log(f"\n[error] {exc}\n")
        finally:
            with STATE_LOCK:
                STATE["running"] = False
                STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                current_proc = None
                stop_requested = False


    threading.Thread(target=worker, daemon=True).start()


def stop_job() -> bool:
    global stop_requested
    with STATE_LOCK:
        if not STATE["running"]:
            return False
        proc = current_proc
        job = STATE["job"]
        stop_requested = True

    append_log("\n[stop requested]\n")
    if proc and proc.poll() is None:
        terminate_process_tree(proc)
    return True


def resolve_under(path: str, base: Path) -> Path:

    target = Path(path).resolve()
    base = base.resolve()
    if target != base and base not in target.parents:
        raise ValueError("路径不在当前打标输出目录内。")
    return target


def parse_label_names(raw: str) -> list[str]:
    labels = [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]
    return labels or ["object"]


def label_image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def write_label_voc_xml(xml_path: Path, image_name: str, frame, objects: list[LabelTrackObject]) -> None:
    height, width = frame.shape[:2]
    depth = frame.shape[2] if len(frame.shape) == 3 else 1
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"
    for obj in objects:
        if not obj.ok or obj.hidden:
            continue
        x, y, w, h = obj.bbox
        item = ET.SubElement(root, "object")
        ET.SubElement(item, "name").text = obj.label
        ET.SubElement(item, "truncated").text = "0"
        ET.SubElement(item, "difficult").text = "0"
        ET.SubElement(item, "occluded").text = "0"
        box = ET.SubElement(item, "bndbox")
        ET.SubElement(box, "xmin").text = str(x)
        ET.SubElement(box, "ymin").text = str(y)
        ET.SubElement(box, "xmax").text = str(x + w)
        ET.SubElement(box, "ymax").text = str(y + h)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="UTF-8", xml_declaration=True)


def label_session_state(session: dict[str, Any]) -> dict[str, Any]:
    frame = session["frame"]
    height, width = frame.shape[:2]
    return {
        "session_id": session["id"],
        "source_type": session["source_type"],
        "frame_index": session["frame_index"],
        "frame_count": session["frame_count"],
        "width": width,
        "height": height,
        "objects": [label_object_data(obj) for obj in session["objects"]],
        "labels": session["labels"],
        "tracker": session["tracker"],
        "saved": session["saved"],
        "interval": session["interval"],
        "processed": session["processed"],
        "max_frames": session["max_frames"],
        "ended": session["ended"],
        "lost": any(not obj.ok and not obj.hidden for obj in session["objects"]),
    }


def get_label_session(session_id: str) -> dict[str, Any]:
    with LABEL_SESSIONS_LOCK:
        session = LABEL_SESSIONS.get(session_id)
    if session is None:
        raise ValueError("标注会话不存在或已结束。")
    return session


def open_label_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame, "default"
    cap.release()
    raise ValueError("摄像头无法打开。请关闭占用程序、检查系统摄像头权限，或尝试其他索引。")


def read_label_image(image_path: Path, flags: int = cv2.IMREAD_COLOR):
    """通过 Python 文件系统读取图片。"""
    try:
        raw = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"图片文件读取失败：{image_path}（{exc}）") from exc
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if frame is None:
        raise ValueError(f"图片解码失败：{image_path}")
    return frame


def start_label_session(values: dict[str, Any]) -> dict[str, Any]:
    validate("label", values)
    source_type = values.get("label_source_type", "video")
    cap = None
    files: Optional[list[Path]] = None
    source_image_path: Optional[Path] = None
    frame_count = 0
    start_frame = max(0, int(values.get("label_start_frame", "0") or 0))
    if source_type == "images":
        files = label_image_files(Path(values["label_images_input_dir"]).resolve())
        if not files:
            raise ValueError("图片集文件夹内没有可标注图片。")
        frame_index = min(start_frame, len(files) - 1)
        source_image_path = files[frame_index]
        frame = read_label_image(source_image_path)
        frame_count = len(files)
    elif source_type == "camera":
        cap, frame, backend = open_label_camera(int(values["label_camera_index"]))
        frame_index = 0
        append_log(f"\n[网页标注] 摄像头 {values['label_camera_index']} 已通过 {backend} 打开。\n")
    else:
        video_path = Path(values["label_video"]).expanduser().resolve()
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("视频文件不存在或格式不支持。")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("视频无法打开，可能是当前 OpenCV 不支持该编码。")
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise ValueError("无法读取视频起始帧。")
        frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    images_dir = Path(values["label_images_dir"]).resolve()
    annotations_dir = Path(values["label_annotations_dir"]).resolve()
    if source_type != "images":
        images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session = {
        "id": session_id, "lock": threading.RLock(), "source_type": source_type, "cap": cap, "files": files,
        "frame": frame, "frame_index": frame_index, "frame_count": frame_count, "source_image_path": source_image_path,
        "objects": [], "next_object_id": 1, "saved": 0, "processed": 0, "ended": False,
        "labels": parse_label_names(values["label_name"]), "tracker": values["label_tracker"],
        "interval": max(1, int(values["label_interval"] or 1)), "max_frames": max(0, int(values["label_max_frames"] or 0)),
        "images_dir": images_dir, "annotations_dir": annotations_dir, "prefix": values["label_prefix"].strip() or "track",
        "jpeg_quality": max(1, min(100, int(values["label_jpeg_quality"] or 95))),
    }
    with LABEL_SESSIONS_LOCK:
        LABEL_SESSIONS[session_id] = session
    append_log(f"[网页标注] 会话已创建：{source_type}，请在页面框选目标。\n")
    return label_session_state(session)


def write_label_image(image_path: Path, frame, jpeg_quality: int) -> None:
    """通过 Python 文件系统写入编码结果。"""
    suffix = image_path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError(f"标注图片编码失败：{image_path.name}")
    try:
        image_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise ValueError(f"标注图片写入失败：{image_path}（{exc}）") from exc


def save_label_session_sample(session: dict[str, Any]) -> bool:
    objects = [obj for obj in session["objects"] if obj.ok and not obj.hidden]
    if not objects:
        return False
    frame = session["frame"]
    source_image_path = session["source_image_path"]
    if source_image_path is not None:
        image_name = source_image_path.name
        xml_path = session["annotations_dir"] / f"{source_image_path.stem}.xml"
    else:
        stem = f"{session['prefix']}_{session['frame_index']:06d}"
        image_name = stem + ".jpg"
        image_path = session["images_dir"] / image_name
        write_label_image(image_path, frame, session["jpeg_quality"])
        xml_path = session["annotations_dir"] / f"{stem}.xml"
    write_label_voc_xml(xml_path, image_name, frame, objects)
    session["saved"] += 1
    return True


def advance_label_session(session: dict[str, Any]) -> dict[str, Any]:
    if session["ended"]:
        return label_session_state(session)
    if session["max_frames"] and session["processed"] >= session["max_frames"]:
        session["ended"] = True
        return label_session_state(session)
    if session["source_type"] == "images":
        next_index = session["frame_index"] + 1
        if next_index >= len(session["files"]):
            session["ended"] = True
            return label_session_state(session)
        frame = read_label_image(session["files"][next_index])
        session["frame_index"] = next_index
        session["source_image_path"] = session["files"][next_index]
    else:
        ok, frame = session["cap"].read()
        if not ok or frame is None:
            session["ended"] = True
            return label_session_state(session)
        if session["source_type"] == "camera":
            session["frame_index"] += 1
        else:
            session["frame_index"] = max(0, int(session["cap"].get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        session["source_image_path"] = None
    session["frame"] = frame
    session["processed"] += 1
    for obj in session["objects"]:
        if not obj.ok or obj.hidden:
            continue
        ok, bbox = obj.tracker.update(frame)
        obj.bbox = sanitize_label_bbox(bbox, frame.shape[1], frame.shape[0])
        obj.ok = bool(ok)
    if session["objects"] and session["frame_index"] % session["interval"] == 0:
        save_label_session_sample(session)
    return label_session_state(session)


def seek_label_session(session: dict[str, Any], frame_index: int) -> dict[str, Any]:
    """Jump a video session to an exact frame without running the tracker."""
    if session["source_type"] != "video":
        raise ValueError("只有视频来源支持拖动帧进度。")
    if session["ended"]:
        raise ValueError("视频会话已经结束，请重新开始标注。")
    if session["frame_count"] <= 0:
        raise ValueError("当前视频没有可用的帧。")
    target = max(0, min(int(frame_index), session["frame_count"] - 1))
    cap = session["cap"]
    if cap is None or not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
        raise ValueError("视频无法跳转到指定帧。")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise ValueError("指定帧读取失败。")
    actual = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    session["frame"] = frame
    session["frame_index"] = actual
    session["source_image_path"] = None
    # Trackers cannot safely resume after a random seek; force manual correction.
    for obj in session["objects"]:
        obj.ok = False
    return label_session_state(session)


def end_label_session(session_id: str) -> None:
    with LABEL_SESSIONS_LOCK:
        session = LABEL_SESSIONS.pop(session_id, None)
    if session is None:
        return
    with session["lock"]:
        cap = session.get("cap")
        if cap is not None:
            cap.release()
        session["ended"] = True
    append_log("[网页标注] 会话已结束。\n")





def parse_voc_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default=xml_path.with_suffix(".jpg").name)
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="")
        box = obj.find("bndbox")
        if box is None:
            continue
        boxes.append({
            "name": name,
            "xmin": int(float(box.findtext("xmin", default="0"))),
            "ymin": int(float(box.findtext("ymin", default="0"))),
            "xmax": int(float(box.findtext("xmax", default="0"))),
            "ymax": int(float(box.findtext("ymax", default="0"))),
        })
    return {"filename": filename, "boxes": boxes}


def label_result_images_dir(values: dict[str, Any]) -> Path:
    if values.get("label_source_type") == "images":
        return Path(values.get("label_images_input_dir", "")).resolve()
    return Path(values["label_images_dir"]).resolve()


def list_label_results(values: dict[str, Any]) -> list[dict[str, Any]]:
    images_dir = label_result_images_dir(values)
    annotations_dir = Path(values["label_annotations_dir"]).resolve()
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return []
    results = []
    xml_files = sorted(annotations_dir.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
    for xml_path in xml_files[:300]:
        try:
            meta = parse_voc_xml(xml_path)
        except Exception:
            continue
        image_path = images_dir / meta["filename"]
        if not image_path.exists():
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = images_dir / (xml_path.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break
        if not image_path.exists():
            continue
        results.append({
            "stem": xml_path.stem,
            "image": str(image_path),
            "xml": str(xml_path),
            "boxes": meta["boxes"],
            "mtime": int(xml_path.stat().st_mtime),
        })
    return results


def list_label_videos(values: dict[str, Any]) -> list[dict[str, Any]]:
    video_dir_raw = values.get("label_video_dir", "").strip()
    if not video_dir_raw:
        return []
    video_dir = Path(video_dir_raw).expanduser().resolve()
    if not video_dir.is_dir():
        return []
    videos = []
    for path in video_dir.rglob("*"):
        if len(videos) >= MAX_LABEL_VIDEOS:
            break
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        videos.append({
            "name": path.name,
            "stem": path.stem,
            "path": str(path),
            "rel": str(path.relative_to(video_dir)),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        })
    videos.sort(key=lambda item: item["rel"].lower())
    return videos





def list_train_plots() -> tuple[Path, list[dict[str, Any]]]:
    with STATE_LOCK:
        plot_dir_raw = str(STATE["markers"].get("train_plot_dir", "")).strip()
    if not plot_dir_raw:
        return Path(), []
    plot_dir = Path(plot_dir_raw).resolve()
    if not plot_dir.is_dir():
        return plot_dir, []
    items = [
        {"name": path.name, "mtime": int(path.stat().st_mtime)}
        for path in sorted(plot_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
    ]
    return plot_dir, items


def send_train_plot(handler: BaseHTTPRequestHandler, plot_dir: Path, name: str) -> None:
    image_path = resolve_under(name, plot_dir)
    if not image_path.is_file() or image_path.suffix.lower() not in TRAIN_IMAGE_EXTENSIONS:
        raise ValueError("训练图片不存在或格式不支持。")
    raw = image_path.read_bytes()
    content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def pick_image_file(initial_path: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前环境不支持系统文件选择器，请手动粘贴图片路径。") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_dir = str(Path(initial_path).expanduser().parent) if initial_path else str(SCRIPT_ROOT)
    if not Path(initial_dir).is_dir():
        initial_dir = str(SCRIPT_ROOT)
    selected = filedialog.askopenfilename(
        parent=root,
        initialdir=initial_dir,
        title="选择测试图片",
        filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")],
    )
    root.destroy()
    return selected


def pick_directory(initial_dir: str = "", title: str = "选择文件夹") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前环境不支持系统文件选择器，请手动粘贴文件夹路径。") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    start = initial_dir if initial_dir and Path(initial_dir).expanduser().exists() else str(SCRIPT_ROOT)
    selected = filedialog.askdirectory(parent=root, initialdir=start, title=title)
    root.destroy()
    return selected


def read_video_preview_frame(video_path: Path):
    backends = [cv2.CAP_ANY]
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append(cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    tried = set()
    for backend in backends:
        if backend in tried:
            continue
        tried.add(backend)
        cap = cv2.VideoCapture(str(video_path), backend)
        try:
            if not cap.isOpened():
                continue
            for frame_no in (0, 1, 3, 10, 30):
                if frame_no:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(30):
                ok = cap.grab()
                if not ok:
                    break
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    return frame
        finally:
            cap.release()
    raise ValueError("无法读取视频预览帧，可能是编码器不受当前 OpenCV 支持。")


def render_video_preview(video_path: Path) -> bytes:
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("视频文件不存在或格式不支持。")
    stat = video_path.stat()
    cache_key = (str(video_path), int(stat.st_mtime), stat.st_size)
    cached = VIDEO_PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        VIDEO_PREVIEW_CACHE.move_to_end(cache_key)
        return cached
    frame = read_video_preview_frame(video_path)
    h, w = frame.shape[:2]
    if w > MAX_VIDEO_PREVIEW_WIDTH:
        scale = MAX_VIDEO_PREVIEW_WIDTH / w
        frame = cv2.resize(frame, (MAX_VIDEO_PREVIEW_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        raise ValueError("视频预览图编码失败。")
    raw = encoded.tobytes()
    VIDEO_PREVIEW_CACHE[cache_key] = raw
    VIDEO_PREVIEW_CACHE.move_to_end(cache_key)
    while len(VIDEO_PREVIEW_CACHE) > VIDEO_PREVIEW_CACHE_LIMIT:
        VIDEO_PREVIEW_CACHE.popitem(last=False)
    return raw






def send_video_file(handler: BaseHTTPRequestHandler, video_path: Path) -> None:
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("视频文件不存在或格式不支持。")
    file_size = video_path.stat().st_size
    start = 0
    end = min(file_size - 1, 8 * 1024 * 1024 - 1)
    status = HTTPStatus.PARTIAL_CONTENT
    range_header = handler.headers.get("Range", "")
    if range_header.startswith("bytes="):
        raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
        left, _, right = raw_range.partition("-")
        if left:
            start = int(left)
            end = int(right) if right else file_size - 1
        elif right:
            suffix_len = int(right)
            start = max(file_size - suffix_len, 0)
            end = file_size - 1
        if start < 0 or end < start or start >= file_size:
            handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            handler.send_header("Content-Range", f"bytes */{file_size}")
            handler.end_headers()
            return
        end = min(end, file_size - 1)
    length = end - start + 1
    content_type = VIDEO_MIME_TYPES.get(video_path.suffix.lower()) or mimetypes.guess_type(str(video_path))[0] or "application/octet-stream"

    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.send_header("Content-Length", str(length))
    handler.end_headers()
    with video_path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            chunk = fh.read(min(256 * 1024, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)








def render_label_preview(image_path: Path, xml_path: Path) -> bytes:
    frame = read_label_image(image_path)
    meta = parse_voc_xml(xml_path)
    line_thickness = max(5, round(min(frame.shape[:2]) / 180))
    text_scale = max(0.7, min(frame.shape[:2]) / 1400)
    for box in meta["boxes"]:
        p1 = (box["xmin"], box["ymin"])
        p2 = (box["xmax"], box["ymax"])
        cv2.rectangle(frame, p1, p2, (40, 220, 120), line_thickness, cv2.LINE_AA)
        label_y = max(24, p1[1] - 8)
        cv2.putText(frame, box["name"], (p1[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), line_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, box["name"], (p1[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (240, 255, 245), max(2, line_thickness // 2), cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("预览图编码失败。")
    return encoded.tobytes()


HTML_PAGE = r'''<!doctype html>

<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>model-training-tool</title>
<style>
:root{--bg:#09111f;--panel:rgba(255,255,255,.08);--panel2:rgba(255,255,255,.13);--text:#eef6ff;--muted:#9fb0c7;--line:rgba(255,255,255,.14);--blue:#56a8ff;--green:#30d287;--purple:#a78bfa;--orange:#ffbd5a;--red:#ff6678;--shadow:0 24px 70px rgba(0,0,0,.35);font-family:"Microsoft YaHei UI","Segoe UI",system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;color:var(--text);background:radial-gradient(circle at 15% 10%,#173c73 0,#09111f 34%),radial-gradient(circle at 86% 0,#41226d 0,transparent 30%),linear-gradient(135deg,#09111f,#0c1528);min-height:100vh}.wrap{width:min(1380px,calc(100% - 36px));margin:0 auto;padding:28px 0 36px}.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;align-items:stretch;margin-bottom:18px}.card{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);border-radius:26px;backdrop-filter:blur(18px)}.title{padding:30px}.eyebrow{display:inline-flex;gap:8px;align-items:center;color:#cde4ff;background:rgba(86,168,255,.13);border:1px solid rgba(86,168,255,.25);padding:7px 12px;border-radius:999px;font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 16px var(--green)}h1{font-size:42px;letter-spacing:-.04em;margin:18px 0 10px}.subtitle{color:var(--muted);line-height:1.8;margin:0;max-width:780px}.guide{padding:24px}.steps{display:grid;gap:12px}.step{display:flex;gap:12px;align-items:flex-start;padding:13px;border:1px solid var(--line);background:rgba(255,255,255,.06);border-radius:18px}.num{flex:0 0 30px;width:30px;height:30px;border-radius:12px;background:linear-gradient(135deg,var(--blue),var(--purple));display:grid;place-items:center;font-weight:800}.step b{display:block;margin-bottom:3px}.step span{color:var(--muted);font-size:13px;line-height:1.5}.layout{display:grid;grid-template-columns:290px 1fr;gap:18px}.side{position:sticky;top:18px;height:fit-content;padding:16px}.nav{display:grid;gap:10px}.nav button{all:unset;cursor:pointer;padding:15px 16px;border-radius:18px;color:var(--muted);border:1px solid transparent;display:flex;justify-content:space-between;align-items:center}.nav button.active{background:linear-gradient(135deg,rgba(86,168,255,.22),rgba(167,139,250,.18));border-color:rgba(255,255,255,.18);color:var(--text)}.status{margin-top:14px;padding:14px;border-radius:18px;background:rgba(0,0,0,.23);border:1px solid var(--line);overflow:hidden}.pill{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border-radius:999px;font-size:13px;border:1px solid var(--line);color:var(--muted);max-width:100%}.pill span:last-child{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pill.run{color:#bff8dc;border-color:rgba(48,210,135,.35);background:rgba(48,210,135,.1)}.pill.idle{color:#d5e5ff;background:rgba(86,168,255,.08)}.main{display:grid;gap:18px}.tab{display:none}.tab.active{display:block}.section{padding:22px;margin-bottom:18px}.section h2{margin:0 0 6px;font-size:24px}.hint{margin:0 0 18px;color:var(--muted);line-height:1.65}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.field{grid-column:span 6}.field.sm{grid-column:span 3}.field.full{grid-column:1/-1}label{display:block;color:#d9e9ff;font-size:13px;margin:0 0 7px}input,select{width:100%;background:rgba(5,12,24,.72);border:1px solid var(--line);border-radius:14px;color:var(--text);padding:12px 13px;outline:none;transition:.2s}input:focus,select:focus{border-color:rgba(86,168,255,.72);box-shadow:0 0 0 4px rgba(86,168,255,.13)}.choice{display:flex;gap:10px;flex-wrap:wrap}.choice label{margin:0;cursor:pointer}.choice input{display:none}.choice span{display:inline-flex;padding:10px 13px;border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.06);color:var(--muted)}.choice input:checked+span{color:var(--text);border-color:rgba(86,168,255,.58);background:rgba(86,168,255,.18)}.actions{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-top:18px}.log-input-actions{justify-content:flex-start;flex-wrap:nowrap}.log-input-actions #job-input{flex:1 1 0;min-width:0}.log-input-actions .job-input-secret-label{display:inline-flex;align-items:center;gap:7px;flex:0 0 auto;margin:0;white-space:nowrap}.log-input-actions input[type="checkbox"]{width:16px;height:16px;margin:0;padding:0;accent-color:var(--blue)}.log-input-actions .btn{flex:0 0 auto}@media(max-width:640px){.log-input-actions{flex-wrap:wrap}.log-input-actions #job-input{flex-basis:100%}.log-input-actions .btn{margin-left:auto}}.btns{display:flex;gap:10px;flex-wrap:wrap}.btn{all:unset;cursor:pointer;border-radius:15px;padding:12px 17px;font-weight:700;border:1px solid var(--line);background:var(--panel2)}.btn.primary{background:linear-gradient(135deg,#238bff,#8b5cf6);border:0}.btn.green{background:linear-gradient(135deg,#18aa69,#22c98a);border:0}.btn.blue{background:linear-gradient(135deg,#1877f2,#56a8ff);border:0}.btn.red{background:rgba(255,102,120,.14);border-color:rgba(255,102,120,.35);color:#ffd7dd}.cmd{margin-top:14px;background:#050b15;border:1px solid var(--line);border-radius:18px;padding:14px;color:#bad4f6;white-space:pre-wrap;word-break:break-all;font-family:"Cascadia Mono",Consolas,monospace;font-size:12px;line-height:1.55}.log{height:360px;overflow:auto;background:#030813;border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:16px;white-space:pre-wrap;color:#c6f6d5;font-family:"Cascadia Mono",Consolas,monospace;font-size:12px;line-height:1.55}.toast{position:fixed;right:22px;bottom:22px;max-width:420px;padding:14px 16px;border-radius:16px;background:#101d33;border:1px solid var(--line);box-shadow:var(--shadow);display:none}.toast.show{display:block}.mini{color:var(--muted);font-size:12px;margin-top:7px}.markers{display:grid;gap:8px;margin-top:10px;min-width:0}.marker{display:grid;grid-template-columns:1fr;gap:5px;align-items:start;padding:10px;border-radius:14px;background:rgba(255,255,255,.05);border:1px solid var(--line);font-size:12px;color:var(--muted);min-width:0;overflow:hidden}.marker b{color:#dcecff;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.marker span{min-width:0;overflow-wrap:anywhere;word-break:break-all;white-space:normal;line-height:1.45}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-top:16px}.sample{overflow:hidden;border-radius:18px;border:1px solid var(--line);background:rgba(255,255,255,.06)}.sample img{width:100%;display:block;aspect-ratio:16/10;object-fit:cover;background:#050b15}.sample .preview-trigger{all:unset;cursor:zoom-in;display:block;width:100%;position:relative}.sample .preview-trigger::after{content:'点击放大';position:absolute;right:10px;bottom:10px;padding:5px 8px;border-radius:999px;background:rgba(3,8,19,.72);border:1px solid rgba(255,255,255,.2);color:#fff;font-size:11px;opacity:0;transform:translateY(4px);transition:.18s}.sample .preview-trigger:hover::after,.sample .preview-trigger:focus-visible::after{opacity:1;transform:translateY(0)}.sample .preview-trigger:focus-visible{outline:2px solid var(--blue);outline-offset:-3px}.sample .meta{padding:11px;font-size:12px;color:var(--muted);display:grid;gap:8px}.sample .delete{all:unset;cursor:pointer;text-align:center;padding:9px 10px;border-radius:12px;background:rgba(255,102,120,.14);border:1px solid rgba(255,102,120,.35);color:#ffd7dd;font-weight:700}.empty{padding:16px;border:1px dashed var(--line);border-radius:18px;color:var(--muted);margin-top:14px}.input-action{display:flex;gap:10px;align-items:stretch}.input-action input{min-width:0;flex:1}.input-action .btn{white-space:nowrap}@media(max-width:520px){.input-action{flex-direction:column}}.label-workspace{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:16px;margin-top:18px}.train-board{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:18px}.progress-card{border:1px solid var(--line);background:rgba(255,255,255,.055);border-radius:22px;padding:16px;overflow:hidden}.progress-card.wide{grid-column:1/-1}.progress-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}.progress-head b{font-size:16px}.progress-head span{color:var(--muted);font-size:12px}.bar{height:16px;border-radius:999px;background:rgba(5,12,24,.72);border:1px solid var(--line);overflow:hidden}.bar div{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--green),var(--blue),var(--purple));transition:width .25s ease}.metrics-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:12px}.metrics-grid div{padding:10px;border-radius:14px;background:rgba(0,0,0,.18);border:1px solid var(--line);min-width:0}.metrics-grid span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.metrics-grid b{display:block;color:#eff8ff;font-size:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.loss-grid{grid-template-columns:repeat(3,1fr)}canvas{width:100%;height:180px;margin-top:10px;border-radius:14px;background:rgba(3,8,19,.72);border:1px solid rgba(255,255,255,.1)}.panel{border:1px solid var(--line);background:rgba(255,255,255,.055);border-radius:22px;padding:16px}.panel h3{margin:0 0 10px;font-size:16px}.queue-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}.count{color:var(--muted);font-size:12px}.video-list{display:grid;gap:8px;max-height:300px;overflow:auto;padding-right:4px}.video-preview{margin-top:14px}.video-preview-box{min-height:220px;border:1px dashed var(--line);border-radius:18px;background:rgba(5,12,24,.42);display:grid;place-items:center;overflow:hidden;color:var(--muted);font-size:12px;text-align:center;position:relative}.video-preview-box.clickable{cursor:pointer;border-style:solid;border-color:rgba(86,168,255,.42)}.video-preview-box img,.video-preview-box video{width:100%;height:100%;display:block;object-fit:contain;background:#050b15}.video-preview-box video{min-height:220px}.play-overlay{position:absolute;inset:0;display:grid;place-items:center;background:linear-gradient(180deg,rgba(5,12,24,.08),rgba(5,12,24,.48));pointer-events:none}.play-button{width:66px;height:66px;border-radius:50%;display:grid;place-items:center;background:rgba(86,168,255,.86);box-shadow:0 14px 36px rgba(0,0,0,.42);color:white;font-size:30px;line-height:1;transform:translateY(-2px)}.video-item{all:unset;cursor:pointer;display:grid;gap:5px;padding:11px 12px;border-radius:15px;border:1px solid var(--line);background:rgba(5,12,24,.42)}.video-item.active{border-color:rgba(86,168,255,.75);background:rgba(86,168,255,.16)}.video-item.done{border-color:rgba(48,210,135,.38)}.video-item b{font-size:13px;color:#edf7ff;word-break:break-all}.video-item span{font-size:12px;color:var(--muted);word-break:break-all}.current-video{display:grid;gap:8px;padding:12px 14px;border-radius:18px;background:rgba(86,168,255,.12);border:1px solid rgba(86,168,255,.24);margin-bottom:14px}.current-video b{word-break:break-all}.label-config{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.label-config .field{grid-column:span 6}.label-config .field.sm{grid-column:span 3}.label-config .field.full{grid-column:1/-1}.quick-help{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.quick-help div{padding:10px;border-radius:14px;background:rgba(0,0,0,.18);border:1px solid var(--line);font-size:12px;color:var(--muted)}.quick-help b{display:block;color:#e6f2ff;margin-bottom:3px}@media(max-width:1100px){.label-workspace{grid-template-columns:1fr}.quick-help{grid-template-columns:repeat(2,1fr)}}@media(max-width:980px){.hero,.layout{grid-template-columns:1fr}.side{position:static}.field,.field.sm,.label-config .field,.label-config .field.sm{grid-column:1/-1}h1{font-size:32px}}.label-studio{margin-top:18px}.label-studio[hidden]{display:none}.label-stage{position:relative;background:#030813;border:1px solid var(--line);border-radius:20px;overflow:hidden;min-height:300px;display:grid;place-items:center}.label-stage img{display:block;width:100%;max-height:650px;object-fit:contain;user-select:none}.label-stage canvas{position:absolute;inset:0;width:100%;height:100%;margin:0;border:0;background:transparent;touch-action:none;cursor:crosshair}.label-stage.empty-stage{color:var(--muted);padding:28px;text-align:center}.label-studio-grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}.label-status{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.label-status span{padding:7px 10px;border-radius:999px;background:rgba(86,168,255,.12);border:1px solid rgba(86,168,255,.24);font-size:12px;color:#dcecff}.label-object-list{display:grid;gap:8px;max-height:360px;overflow:auto}.label-object{cursor:pointer;display:grid;grid-template-columns:28px minmax(0,1fr);gap:8px;align-items:center;padding:8px 11px;border:1px solid var(--line);border-radius:14px;background:rgba(5,12,24,.42);font-size:12px}.label-object-info{display:grid;gap:4px;min-width:0}.label-object-info b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.label-object.active{border-color:rgba(86,168,255,.8);background:rgba(86,168,255,.17)}.label-object.lost{border-color:rgba(255,102,120,.55);color:#ffd7dd}.label-object span{color:var(--muted)}.label-object-visibility{all:unset;cursor:pointer;width:28px;height:28px;display:grid;place-items:center;border-radius:8px;color:#cfe3fb}.label-object-visibility:hover{background:rgba(255,255,255,.12)}.label-object-visibility svg{width:17px;height:17px;fill:none;stroke:currentColor;stroke-width:1.9;stroke-linecap:round;stroke-linejoin:round}.label-object-visibility.is-hidden{color:var(--muted)}.label-object-visibility.is-hidden svg{opacity:.58}.image-lightbox{position:fixed;z-index:1000;inset:0;padding:28px;display:grid;place-items:center;background:rgba(2,6,14,.9);backdrop-filter:blur(10px)}.image-lightbox[hidden]{display:none}.image-lightbox-content{position:relative;display:grid;gap:10px;max-width:min(1400px,100%);max-height:100%;width:max-content}.image-lightbox img{display:block;max-width:calc(100vw - 56px);max-height:calc(100vh - 110px);object-fit:contain;border-radius:14px;border:1px solid rgba(255,255,255,.2);box-shadow:0 24px 80px rgba(0,0,0,.5);background:#050b15}.image-lightbox-title{color:#dcecff;font-size:13px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 48px}.image-lightbox-close{all:unset;cursor:pointer;position:absolute;right:0;top:0;width:38px;height:38px;display:grid;place-items:center;border-radius:50%;background:rgba(3,8,19,.82);border:1px solid rgba(255,255,255,.23);color:#fff;font-size:27px;line-height:1;z-index:1}.image-lightbox-close:hover{background:rgba(255,102,120,.3)}.label-help{font-size:12px;color:var(--muted);line-height:1.65;margin-top:12px}@media(max-width:980px){.label-studio-grid{grid-template-columns:1fr}}
.label-timeline{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center;margin-top:12px;padding:10px 12px;border:1px solid var(--line);border-radius:14px;background:rgba(5,12,24,.42)}.label-timeline label{margin:0;white-space:nowrap}.label-timeline input[type=range]{width:100%;padding:0;accent-color:var(--blue);cursor:ew-resize}.label-timeline input[type=range]:disabled{opacity:.45;cursor:not-allowed}.label-timeline span{color:var(--muted);font-size:12px;white-space:nowrap}
.label-choice-dialog{position:fixed;z-index:1100;inset:0;display:grid;place-items:center;padding:22px;background:rgba(2,6,14,.78);backdrop-filter:blur(8px)}.label-choice-dialog[hidden]{display:none}.label-choice-content{width:min(440px,100%);padding:22px;border:1px solid rgba(255,255,255,.2);border-radius:20px;background:#101d33;box-shadow:0 24px 80px rgba(0,0,0,.55)}.label-choice-content h3{margin:0 0 7px;font-size:18px}.label-choice-content p{margin:0 0 16px;color:var(--muted);font-size:13px}.label-choice-options{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}.label-choice-option{all:unset;cursor:pointer;display:flex;align-items:center;gap:10px;min-height:44px;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.06);color:var(--text);font-weight:700}.label-choice-option:hover,.label-choice-option:focus-visible{border-color:rgba(86,168,255,.8);background:rgba(86,168,255,.2);outline:none}.label-choice-option-key{display:grid;place-items:center;flex:0 0 24px;width:24px;height:24px;border-radius:7px;background:rgba(86,168,255,.2);color:#dcecff;font-size:12px}.label-choice-cancel{width:100%;text-align:center}
</style>
<style>
#label-stage #label-frame-image { pointer-events: none; -webkit-user-drag: none; }
#label-stage #label-frame-canvas { right: auto; bottom: auto; user-select: none; }
</style>
</head>
<body>
<div id="image-lightbox" class="image-lightbox" hidden role="dialog" aria-modal="true" aria-label="标注图片预览"><div class="image-lightbox-content"><button id="image-lightbox-close" class="image-lightbox-close" type="button" aria-label="关闭图片预览" title="关闭（Esc）">×</button><img id="image-lightbox-image" alt="标注结果放大预览"><div id="image-lightbox-title" class="image-lightbox-title"></div></div></div>
<div id="label-choice-dialog" class="label-choice-dialog" hidden role="dialog" aria-modal="true" aria-labelledby="label-choice-title"><div class="label-choice-content"><h3 id="label-choice-title">选择目标标签</h3><p>请选择刚刚绘制的标注框所属类别。</p><div id="label-choice-options" class="label-choice-options"></div><button id="label-choice-cancel" class="btn" type="button">取消</button></div></div>
<div class="wrap">
  <div class="hero">
    <div class="card title">
      <div class="eyebrow"><span class="dot"></span> Web 面板运行于 8989 端口</div>

      <h1>model-training-tool</h1>
      <p class="subtitle">按“准备数据 → 训练模型 → 测试效果”的顺序完成流程。</p>
    </div>
    <div class="card guide">
      <div class="steps">
        <div class="step"><div class="num">1</div><div><b>确认数据集</b><span>分别填写图片目录、XML 标注目录和输出根目录。</span></div></div>
        <div class="step"><div class="num">2</div><div><b>配置本机训练</b><span>选择数据目录、模型和训练参数。</span></div></div>
        <div class="step"><div class="num">3</div><div><b>测试效果</b><span>使用训练生成的 .pt 模型进行摄像头、单图或图片目录推理。</span></div></div>
      </div>
    </div>
  </div>

  <div class="layout">
    <aside class="card side">
      <div class="nav">
        <button data-tab="train" class="active">训练配置 <span>01</span></button>
        <button data-tab="test">模型测试 <span>02</span></button>
        <button data-tab="label">视频打标 <span>03</span></button>
        <button data-tab="logs">运行日志 <span>04</span></button>

      </div>
      <div class="status">
        <div id="runPill" class="pill idle"><span class="dot"></span><span>空闲</span></div>
        <div class="mini" id="jobInfo">暂无任务</div>
        <div class="markers" id="markers"></div>
      </div>
    </aside>

    <main class="main">
      <section id="tab-train" class="tab active card section">
        <h2>训练配置</h2><p class="hint">训练任务在当前 Linux 环境中执行。</p>
        <div class="grid">
          <div class="field full"><label>训练任务</label><div class="choice"><label><input name="train_task" type="radio" value="detect"><span>目标检测（图片 + XML 标注）</span></label><label><input name="train_task" type="radio" value="classify"><span>图像分类（类别子文件夹）</span></label></div><div class="mini" id="train-task-hint">检测：Images Dir 与 Annotations Dir 一一对应；分类：Images Dir 下每个子文件夹即一个类别，例如 normal/、defect/。</div></div>
          <div class="field full"><label>Dataset Root / Output Root</label><input id="dataset_root"></div>
          <div class="field"><label>Images Dir</label><input id="train_images_dir" placeholder="例如 E:/dataset/images"></div>
          <div class="field" id="annotations-field"><label>Annotations Dir</label><input id="train_annotations_dir" placeholder="例如 E:/dataset/annotations"></div>
          <div class="field full"><label>训练集 / 验证集比例：<span id="split_ratio_text">训练 80% / 验证 20%</span></label><input id="train_ratio_percent" type="range" min="1" max="100" step="1"><div class="mini">拖动条表示训练集占比，验证集占比自动为剩余比例；建议常用 80% / 20%。</div></div>
          <div class="field"><label>Base Model</label><input id="base_model"></div>

          <div class="field"><label>Conda Env</label><input id="conda_env"></div>
          <div class="field"><label>PyTorch CUDA</label><select id="torch_cuda"><option value="cu128">CUDA 12.8 (RTX 50 系列/默认)</option><option value="cu124">CUDA 12.4</option><option value="cu121">CUDA 12.1</option><option value="cu118">CUDA 11.8</option><option value="cpu">CPU 版 PyTorch</option><option value="none">不自动安装/更新 PyTorch</option></select></div>
          <div class="field"><label>Train Device</label><div class="choice"><label><input name="train_device" type="radio" value="cuda"><span>CUDA/GPU</span></label><label><input name="train_device" type="radio" value="cpu"><span>CPU</span></label></div></div>
          <div class="field"><label>Cache</label><select id="train_cache"><option value="False">关闭 cache（默认）</option><option value="True">开启 cache 到内存</option><option value="disk">开启 cache 到磁盘</option></select><div class="mini">显存/内存足够时可加速训练；数据集较大建议选择 disk 或关闭。</div></div>
          <div class="field"><label>Project Name</label><input id="project_name"></div>

          <div class="field"><label>Model Name</label><input id="model_name"></div>
          <div class="field sm"><label>图片宽度</label><input id="img_width" type="number" min="32" step="32" inputmode="numeric"></div>
          <div class="field sm"><label>图片高度</label><input id="img_height" type="number" min="32" step="32" inputmode="numeric"></div>
          <div class="field"><label>图片适配方式</label><select id="image_resize_mode"><option value="crop">裁剪（居中裁剪后缩放）</option><option value="letterbox">等比缩放（留边填充）</option><option value="stretch">拉伸（直接缩放）</option></select><div class="mini">会在生成训练集时同步变换图片与标注框；推荐使用等比缩放。</div></div>
          <div class="field sm"><label>Epochs</label><input id="epochs"></div>
          <div class="field sm"><label>Batch</label><input id="batch"></div>
          <div class="field sm"><label>Lr0</label><input id="lr0"></div>
        </div>
        <div class="train-board">
          <div class="progress-card wide">
            <div class="progress-head"><b>资源预估</b><span id="resource-note">配置变化后自动估算</span></div>
            <div class="metrics-grid">
              <div><span>训练图片</span><b id="estimate-images">-</b></div>
              <div><span>内存</span><b id="estimate-ram">-</b></div>
              <div><span>显存</span><b id="estimate-vram">-</b></div>
              <div><span>Cache 额外</span><b id="estimate-cache">-</b></div>
              <div><span>图片宽 × 高</span><b id="estimate-imgsz">-</b></div>
              <div><span>Batch</span><b id="estimate-batch">-</b></div>
            </div>
            <div class="mini" id="resource-detail">估算值仅供参考，实际峰值会随模型、增强策略、驱动和环境波动。</div>
          </div>
        </div>
        <div class="actions"><div class="btns"><button class="btn" onclick="copyCommand('train')">复制训练命令</button><button class="btn" onclick="saveDefaults('训练配置')">存为默认</button></div><button class="btn green" onclick="runAction('train')">开始训练</button></div>
        <div class="train-board">

          <div class="progress-card wide">
            <div class="progress-head"><b>训练进度</b><span id="train-phase">等待开始</span></div>
            <div class="bar"><div id="epoch-bar" style="width:0%"></div></div>
            <div class="metrics-grid">
              <div><span>Epoch</span><b id="epoch-text">-</b></div>
              <div><span>Batch</span><b id="batch-text">-</b></div>
              <div><span>GPU</span><b id="gpu-text">-</b></div>
              <div><span>速度</span><b id="speed-text">-</b></div>
              <div><span>耗时</span><b id="elapsed-text">-</b></div>
              <div><span>剩余</span><b id="eta-text">-</b></div>
            </div>
          </div>
          <div class="progress-card">
            <div class="progress-head"><b>Loss</b><span id="loss-title">训练损失</span></div>
            <div class="metrics-grid loss-grid">
              <div id="loss-item"><span id="loss-label">loss</span><b id="loss-value">-</b></div>
              <div id="box-loss-item"><span>box</span><b id="box-loss">-</b></div>
              <div id="cls-loss-item"><span>cls</span><b id="cls-loss">-</b></div>
              <div id="dfl-loss-item"><span>dfl</span><b id="dfl-loss">-</b></div>
            </div>
            <canvas id="loss-chart" width="520" height="180"></canvas>
          </div>
          <div class="progress-card">
            <div class="progress-head"><b>验证指标</b><span id="val-text">-</span></div>
            <div id="detect-metrics" class="metrics-grid loss-grid">
              <div><span>Precision</span><b id="precision-text">-</b></div>
              <div><span>Recall</span><b id="recall-text">-</b></div>
              <div><span>mAP50</span><b id="map50-text">-</b></div>
              <div><span>mAP50-95</span><b id="map5095-text">-</b></div>
            </div>
            <div id="classify-metrics" class="metrics-grid loss-grid" hidden>
              <div><span>Top-1 Accuracy</span><b id="top1-text">-</b></div>
              <div><span>Top-5 Accuracy</span><b id="top5-text">-</b></div>
            </div>
            <canvas id="metric-chart" width="520" height="180"></canvas>
          </div>
        </div>
        <div class="cmd" id="cmd-train"></div>
      </section>

      <section id="tab-test" class="tab card section">
        <h2>模型测试</h2><p class="hint">自动兼容目标检测与图像分类 `.pt` 模型。单张图片会保存带标注的预测图；文件夹测试会递归处理所有图片并输出预测图与 `predictions.csv`。</p>
        <div class="grid">
          <div class="field full"><label>Test Model .pt</label><input id="test_model"></div>
          <div class="field full"><label>Source</label><div class="choice"><label><input name="test_source" type="radio" value="camera"><span>Camera</span></label><label><input name="test_source" type="radio" value="image"><span>单张图片</span></label><label><input name="test_source" type="radio" value="folder"><span>图片文件夹</span></label></div></div>
          <div class="field full"><label>图片路径</label><div class="input-action"><input id="test_image_file" placeholder="选择单张图片时填写"><button class="btn" onclick="pickTestImage()">选择图片</button></div></div>
          <div class="field full"><label>图片文件夹路径</label><div class="input-action"><input id="test_image_folder" placeholder="选择图片文件夹时填写，支持递归扫描"><button class="btn" onclick="pickTestImageFolder()">选择文件夹</button></div></div>
          <div class="field full"><label>批量测试输出文件夹（可选）</label><div class="input-action"><input id="test_output_dir" placeholder="留空时输出到图片文件夹/predict_results"><button class="btn" onclick="pickTestOutputDir()">选择文件夹</button></div></div>
          <div class="field"><label>Camera Index</label><input id="camera_index"></div>
          <div class="field"><label>Confidence（仅检测模型生效）</label><input id="conf"></div>
        </div>
        <div class="actions"><button class="btn" onclick="copyCommand('test')">复制测试命令</button><button class="btn primary" onclick="runAction('test')">开始测试</button></div>
        <div class="cmd" id="cmd-test"></div>
      </section>

      <section id="tab-label" class="tab card section">
        <h2>自动跟踪标注</h2><p class="hint">支持连续视频或按文件名排序的图片集（适用于自行提取、筛选的视频帧）。图片集模式直接为原图生成同名 XML，不会复制或重编码图片；应保持图片顺序与视频时间顺序一致。</p>
        <div class="grid">
          <div class="field full"><label>标注来源</label><div class="choice"><label><input name="label_source_type" type="radio" value="video"><span>视频</span></label><label><input name="label_source_type" type="radio" value="camera"><span>摄像头</span></label><label><input name="label_source_type" type="radio" value="images"><span>图片集</span></label></div></div>
        </div>
        <div id="label-video-source" class="label-source">
          <div class="grid">
            <div class="field full"><label>Video Folder</label><input id="label_video_dir" placeholder="例如 D:/videos 或 E:/datasets/raw_videos"></div>
          </div>
          <div class="actions"><div class="btns"><button class="btn" onclick="pickLabelVideoDir()">选择视频文件夹</button><button class="btn blue" onclick="loadLabelVideos()">读取文件夹视频</button></div><div class="btns"><button class="btn" onclick="selectPrevVideo()">上一个</button><button class="btn" onclick="selectNextVideo()">下一个</button></div></div>
          <div class="label-workspace">
            <div class="panel">
              <div class="queue-head"><h3>视频队列</h3><span class="count" id="label-video-count">0 个视频</span></div>
              <div id="label-video-list" class="video-list"><div class="empty">填写 Video Folder 后点击“读取文件夹视频”。</div></div>
              <div class="video-preview">
                <div class="queue-head"><h3>视频预览</h3><span class="count" id="label-preview-name">未选择</span></div>
                <div id="label-video-preview" class="video-preview-box"><span>选择左侧视频后显示首帧预览</span></div>
              </div>
            </div>
            <div class="panel" id="label-video-config">
              <h3>当前视频与标注参数</h3>
              <div class="current-video"><span class="count">当前待标注视频</span><b id="label-current-video">未选择视频</b></div>
              <div class="label-config">
                <div class="field full"><label>Video Path</label><input id="label_video" placeholder="从队列选择，或手动输入视频文件路径"></div>
                <div class="field"><label>Labels</label><input id="label_name" placeholder="多个标签用英文逗号分隔，如 w,person,car"></div>
                <div class="field"><label>Filename Prefix</label><input id="label_prefix"></div>
                <div class="field"><label>Images Dir</label><input id="label_images_dir"></div>
                <div class="field"><label>Annotations Dir</label><input id="label_annotations_dir"></div>
                <div class="field"><label>Tracker</label><select id="label_tracker"><option value="csrt">CSRT</option><option value="kcf">KCF</option><option value="mosse">MOSSE</option><option value="mil">MIL</option><option value="template">Template</option><option value="multi_template">Multi-template（多角度）</option></select></div>

                <div class="field sm"><label>Save Every N Frames</label><input id="label_interval"></div>
                <div class="field sm"><label>Start Frame</label><input id="label_start_frame"></div>
                <div class="field sm"><label>Max Frames</label><input id="label_max_frames"></div>
                <div class="field sm"><label>Display Scale</label><input id="label_display_scale"></div>
                <div class="field sm"><label>JPEG Quality</label><input id="label_jpeg_quality"></div>
              </div>
            </div>
          </div>
        </div>
        <div id="label-camera-source" class="label-source" hidden>
          <div class="panel">
            <h3>摄像头实时标注</h3>
            <p class="hint">启动后会打开本机 OpenCV 标注窗口。确认实时画面后框选目标并选择类别，跟踪过程中会按保存间隔写入 JPEG 和同名 VOC XML；按 q 或 Esc 结束采集。</p>
            <div class="label-config">
              <div class="field"><label>Camera Index</label><input id="label_camera_index" placeholder="0"></div>
              <div class="field"><label>Labels</label><input id="label_name_camera" placeholder="多个标签用英文逗号分隔，如 w,person,car"></div>
              <div class="field"><label>Filename Prefix</label><input id="label_prefix_camera" placeholder="camera"></div>
              <div class="field"><label>Images Dir</label><input id="label_images_dir_camera"></div>
              <div class="field"><label>Annotations Dir</label><input id="label_annotations_dir_camera"></div>
              <div class="field"><label>Tracker</label><select id="label_tracker_camera"><option value="csrt">CSRT</option><option value="kcf">KCF</option><option value="mosse">MOSSE</option><option value="mil">MIL</option><option value="template">Template</option><option value="multi_template">Multi-template（多角度）</option></select></div>
              <div class="field sm"><label>Save Every N Frames</label><input id="label_interval_camera"></div>
              <div class="field sm"><label>Max Frames</label><input id="label_max_frames_camera" placeholder="0 为持续采集"></div>
              <div class="field sm"><label>Display Scale</label><input id="label_display_scale_camera"></div>
              <div class="field sm"><label>JPEG Quality</label><input id="label_jpeg_quality_camera"></div>
            </div>
          </div>
        </div>
        <div id="label-images-source" class="label-source" hidden>
          <div class="panel">
            <h3>图片集与标注参数</h3>
            <p class="hint">选择已按时间顺序命名的帧图片文件夹。将按文件名排序跟踪，原图保留不动，仅在标注目录生成同名 XML。</p>
            <div class="label-config">
              <div class="field full"><label>图片集文件夹</label><div class="input-action"><input id="label_images_input_dir" placeholder="例如 E:/datasets/selected_frames（支持 jpg、png、bmp、webp）"><button class="btn" onclick="pickLabelImagesDir()">选择文件夹</button></div></div>
              <div class="field"><label>Labels</label><input id="label_name_images" placeholder="多个标签用英文逗号分隔，如 w,person,car"></div>
              <div class="field"><label>Annotations Dir</label><input id="label_annotations_dir_images"></div>
              <div class="field"><label>Tracker</label><select id="label_tracker_images"><option value="csrt">CSRT</option><option value="kcf">KCF</option><option value="mosse">MOSSE</option><option value="mil">MIL</option><option value="template">Template</option><option value="multi_template">Multi-template（多角度）</option></select></div>
              <div class="field sm"><label>每 N 张保存</label><input id="label_interval_images"></div>
              <div class="field sm"><label>起始图片序号</label><input id="label_start_frame_images"></div>
              <div class="field sm"><label>最多处理图片</label><input id="label_max_frames_images"></div>
              <div class="field sm"><label>显示缩放</label><input id="label_display_scale_images"></div>
            </div>
            <div class="quick-help"><div><b>1. 选择图片集</b>选择已筛选并按时间顺序命名的帧图片。</div><div><b>2. 开始标注</b>在首张图片框选目标并选择类别。</div><div><b>3. 修正跟踪</b>目标丢失后按 r 修正当前框。</div><div><b>4. 保存结果</b>每张原图对应同名 XML，原图不会被删除。</div></div>
          </div>
        </div>
        <div id="label-browser-studio" class="label-studio panel" hidden>
          <div class="queue-head"><div><h3>网页标注工作台</h3><span class="count" id="label-session-tip">创建会话后，在画面上拖动鼠标框选目标。</span></div><button class="btn red" onclick="endBrowserLabelSession()">结束标注</button></div>
          <div class="label-studio-grid">
            <div>
              <div id="label-stage" class="label-stage empty-stage"><span>选择来源并点击“在网页中开始标注”后，此处显示实时画面。</span><img id="label-frame-image" hidden draggable="false" alt="当前标注画面"><canvas id="label-frame-canvas" hidden></canvas></div>
              <div class="label-timeline"><label for="label-frame-seek">视频位置</label><input id="label-frame-seek" type="range" min="0" max="0" value="0" disabled aria-label="拖动选择视频帧"><span id="label-frame-seek-value">帧 -</span></div>
              <div id="label-session-status" class="label-status"></div>
              <div class="actions"><div class="btns"><button class="btn green" id="label-play-button" onclick="toggleBrowserLabelPlay()">播放跟踪</button><button class="btn" onclick="advanceBrowserLabelFrame()">下一帧</button><button class="btn" onclick="saveBrowserLabelFrame()">保存当前帧</button></div><div class="btns"><button class="btn" onclick="setLabelDrawMode('add')">添加框</button><button class="btn" onclick="setLabelDrawMode('edit')">修正选中框</button><button class="btn" onclick="setLabelDrawMode('sample')">追加选中目标视角</button><button class="btn red" onclick="deleteBrowserLabelObject()">删除选中框</button></div></div>
              <div class="label-help">多角度模式：选择 Multi-template，暂停到新角度后选中目标，点击“追加选中目标视角”并重画该目标。它会保留已有参考图；“修正选中框”会重置参考图。</div>
            </div>
            <div class="panel"><h3>当前目标</h3><div id="label-object-list" class="label-object-list"><div class="empty">尚未框选目标。</div></div></div>
          </div>
        </div>
        <div class="actions"><div class="btns"><button class="btn" onclick="copyCommand('label')">复制当前命令</button><button class="btn" onclick="saveDefaults('自动跟踪标注')">存为默认</button></div><div class="btns"><button class="btn" onclick="loadLabelResults()">刷新标注结果</button><button class="btn primary" id="label-start-button" onclick="runLabelCurrent()">在网页中开始标注</button></div></div>
        <div class="cmd" id="cmd-label"></div>
        <h2 style="margin-top:24px">标注结果</h2><p class="hint">这里显示当前标注样本。视频模式下“删除废图”会同时删除导出图片和 XML；图片集模式下仅删除 XML，保留用户原始图片。</p>
        <div id="label-results" class="gallery"></div>

      </section>


      <section id="tab-logs" class="tab card section">
        <h2>运行日志</h2><p class="hint">这里实时显示训练、模型测试和视频打标输出。</p>
        <div class="actions"><div class="btns"><button class="btn" onclick="refreshState()">刷新</button><button class="btn" onclick="copyLogs()">复制日志</button></div><button class="btn red" onclick="stopJob()">停止当前任务</button></div>
        <div class="log" id="log"></div>
      </section>

    </main>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
const fields=['dataset_root','train_task','train_images_dir','train_annotations_dir','train_ratio_percent','img_width','img_height','image_resize_mode','epochs','batch','lr0','conda_env','base_model','torch_cuda','train_cache','project_name','model_name','test_model','test_image_file','test_image_folder','test_output_dir','camera_index','conf','label_video_dir','label_video','label_camera_index','label_source_type','label_images_input_dir','label_name','label_interval','label_images_dir','label_annotations_dir','label_prefix','label_tracker','label_start_frame','label_max_frames','label_display_scale','label_jpeg_quality'];



let values={};
let labelVideos=[];
let labelVideoIndex=-1;
let labelVisibleCount=150;
const LABEL_VIDEO_PAGE_SIZE=150;
let labelPreviewToken=0;
let labelResultsTimer=null;
let resourceEstimateTimer=null;
let lastResourceEstimateKey='';
let deleteConfirmUntil=0;


const rawLabelPrefix={manual:false,value:''};
function toast(msg){const el=document.getElementById('toast');el.textContent=msg;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2600)}
function updateSplitRatio(){const el=document.getElementById('train_ratio_percent'); const text=document.getElementById('split_ratio_text'); if(!el||!text) return; let train=Math.round(Number(el.value||80)); if(!Number.isFinite(train)) train=80; train=Math.max(1,Math.min(100,train)); if(String(train)!==el.value) el.value=String(train); text.textContent=`训练 ${train}% / 验证 ${100-train}%`}
function fmt(v,d=3){return v===null||v===undefined||v===''||Number.isNaN(Number(v))?'-':Number(v).toFixed(d).replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1')}
function setText(id,value){const el=document.getElementById(id); if(el) el.textContent=value}
function drawChart(id,history,series){const canvas=document.getElementById(id); if(!canvas) return; const ctx=canvas.getContext('2d'); const w=canvas.width,h=canvas.height; ctx.clearRect(0,0,w,h); ctx.fillStyle='rgba(3,8,19,.72)'; ctx.fillRect(0,0,w,h); const rows=(history||[]).filter(x=>series.some(s=>x[s.key]!==null&&x[s.key]!==undefined)); ctx.strokeStyle='rgba(255,255,255,.09)'; ctx.lineWidth=1; for(let i=1;i<4;i++){const y=i*h/4; ctx.beginPath(); ctx.moveTo(34,y); ctx.lineTo(w-10,y); ctx.stroke()} if(rows.length<2){ctx.fillStyle='rgba(159,176,199,.9)'; ctx.font='13px sans-serif'; ctx.fillText('等待更多 epoch 数据...',18,34); return} let vals=[]; for(const r of rows){for(const s of series){const v=Number(r[s.key]); if(Number.isFinite(v)) vals.push(v)}} let min=Math.min(...vals),max=Math.max(...vals); if(min===max){min-=1;max+=1} const pad=(max-min)*0.08; min-=pad; max+=pad; const xOf=i=>34+i*(w-48)/Math.max(1,rows.length-1); const yOf=v=>h-22-(Number(v)-min)*(h-42)/(max-min); ctx.font='11px sans-serif'; series.forEach((s,si)=>{ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath(); let started=false; rows.forEach((r,i)=>{const v=Number(r[s.key]); if(!Number.isFinite(v)) return; const x=xOf(i),y=yOf(v); if(!started){ctx.moveTo(x,y); started=true}else ctx.lineTo(x,y)}); ctx.stroke(); ctx.fillStyle=s.color; ctx.fillText(s.label,38+si*82,16)}); ctx.fillStyle='rgba(159,176,199,.8)'; ctx.fillText(`E${rows[0].epoch}`,34,h-7); ctx.fillText(`E${rows[rows.length-1].epoch}`,w-52,h-7)}
function updateTrainProgress(p){p=p||{}; const task=p.task||document.querySelector('input[name="train_task"]:checked')?.value||'detect'; const classify=task==='classify'; const pct=Math.max(0,Math.min(100,Number(p.percent||0))); const totalEpochs=Number(p.total_epochs||0); const epoch=Number(p.epoch||0); const totalBatches=Number(p.total_batches||0); const batch=Number(p.batch||0); const phaseMap={idle:'等待开始',pending:'准备训练',train:'训练中',val:'验证中',metrics:'指标已更新',export:'正在停止训练并导出'}; setText('train-phase',`${phaseMap[p.phase]||p.phase||'等待开始'}${p.updated_at?' · '+p.updated_at:''}`); const bar=document.getElementById('epoch-bar'); if(bar) bar.style.width=pct+'%'; setText('epoch-text',epoch&&totalEpochs?`${epoch}/${totalEpochs}`:'-'); setText('batch-text',totalBatches?`${batch}/${totalBatches} (${fmt(pct,1)}%)`:'-'); setText('gpu-text',p.gpu_mem||'-'); setText('speed-text',p.speed||'-'); setText('elapsed-text',p.elapsed||'-'); setText('eta-text',p.eta||'-'); const lossItem=document.getElementById('loss-item'); const boxItem=document.getElementById('box-loss-item'); const clsItem=document.getElementById('cls-loss-item'); const dflItem=document.getElementById('dfl-loss-item'); if(lossItem) lossItem.hidden=!classify; if(boxItem) boxItem.hidden=classify; if(clsItem) clsItem.hidden=classify; if(dflItem) dflItem.hidden=classify; setText('loss-title',classify?'分类训练损失':'检测训练损失'); setText('loss-value',fmt(p.loss)); setText('box-loss',fmt(p.box_loss)); setText('cls-loss',fmt(p.cls_loss)); setText('dfl-loss',fmt(p.dfl_loss)); const detectMetrics=document.getElementById('detect-metrics'); const classifyMetrics=document.getElementById('classify-metrics'); if(detectMetrics) detectMetrics.hidden=classify; if(classifyMetrics) classifyMetrics.hidden=!classify; const m=p.metrics||{}; setText('val-text',p.val_total?`${p.val_batch||0}/${p.val_total} (${fmt(p.val_percent||0,1)}%)`:'-'); setText('precision-text',fmt(m.precision)); setText('recall-text',fmt(m.recall)); setText('map50-text',fmt(m.map50)); setText('map5095-text',fmt(m.map50_95)); setText('top1-text',fmt(m.top1_acc)); setText('top5-text',fmt(m.top5_acc)); const lossSeries=classify?[{key:'loss',label:'loss',color:'#ffbd5a'}]:[{key:'box_loss',label:'box',color:'#56a8ff'},{key:'cls_loss',label:'cls',color:'#ffbd5a'},{key:'dfl_loss',label:'dfl',color:'#a78bfa'}]; const metricSeries=classify?[{key:'top1_acc',label:'Top-1',color:'#30d287'},{key:'top5_acc',label:'Top-5',color:'#56a8ff'}]:[{key:'precision',label:'P',color:'#30d287'},{key:'recall',label:'R',color:'#56a8ff'},{key:'map50',label:'mAP50',color:'#ffbd5a'},{key:'map50_95',label:'mAP50-95',color:'#a78bfa'}]; drawChart('loss-chart',p.history||[],lossSeries); drawChart('metric-chart',p.history||[],metricSeries)}
function updateTrainTaskUI(){const task=document.querySelector('input[name="train_task"]:checked')?.value||'detect'; const annotations=document.getElementById('annotations-field'); const hint=document.getElementById('train-task-hint'); if(annotations) annotations.hidden=task==='classify'; if(hint) hint.textContent=task==='classify'?'分类数据集结构：Images Dir/类别名/图片。每个类别至少 2 张图片；Annotations Dir 不参与分类训练；Base Model 请使用分类权重，例如 yolo11n-cls.pt。':'检测数据集结构：Images Dir 与 Annotations Dir 中的同名图片、XML 一一对应。'}
function syncLabelFields(prefix,toCanonical){const pairs=prefix==='camera'?[['label_name_camera','label_name'],['label_prefix_camera','label_prefix'],['label_images_dir_camera','label_images_dir'],['label_annotations_dir_camera','label_annotations_dir'],['label_tracker_camera','label_tracker'],['label_interval_camera','label_interval'],['label_max_frames_camera','label_max_frames'],['label_display_scale_camera','label_display_scale'],['label_jpeg_quality_camera','label_jpeg_quality']]:[['label_name_images','label_name'],['label_annotations_dir_images','label_annotations_dir'],['label_tracker_images','label_tracker'],['label_interval_images','label_interval'],['label_start_frame_images','label_start_frame'],['label_max_frames_images','label_max_frames'],['label_display_scale_images','label_display_scale']]; for(const [sourceId,canonicalId] of pairs){const from=document.getElementById(toCanonical?sourceId:canonicalId); const to=document.getElementById(toCanonical?canonicalId:sourceId); if(from&&to) to.value=from.value}}
function updateLabelSourceUI(){const source=document.querySelector('input[name="label_source_type"]:checked')?.value||'video'; const video=document.getElementById('label-video-source'); const camera=document.getElementById('label-camera-source'); const images=document.getElementById('label-images-source'); const start=document.getElementById('label-start-button'); if(video) video.hidden=source!=='video'; if(camera) camera.hidden=source!=='camera'; if(images) images.hidden=source!=='images'; if(source==='camera') syncLabelFields('camera',false); if(source==='images') syncLabelFields('images',false); if(start) start.textContent=source==='camera'?'在网页中开始摄像头标注':source==='images'?'在网页中开始图片集标注':'在网页中开始视频标注'}
function collect(){const source=document.querySelector('input[name="label_source_type"]:checked')?.value||'video'; if(source==='camera') syncLabelFields('camera',true); if(source==='images') syncLabelFields('images',true); for(const id of fields){const el=document.getElementById(id); if(el) values[id]=el.value} for(const n of ['train_device','train_task','test_source','label_source_type']){const el=document.querySelector(`input[name="${n}"]:checked`); if(el) values[n]=el.value} return values}
function resourceEstimateKey(v){return [v.train_task||'detect',v.train_images_dir||'',v.base_model||'',v.img_width||'',v.img_height||'',v.image_resize_mode||'',v.batch||'',v.train_cache||'',v.train_device||''].join('|')}
function showResourceEstimate(e){e=e||{}; setText('estimate-images',e.image_count!==undefined?`${e.image_count} 张`:'-'); setText('estimate-ram',e.ram_text||'-'); setText('estimate-vram',e.vram_text||'-'); setText('estimate-cache',e.cache_text||'-'); setText('estimate-imgsz',e.img_size||'-'); setText('estimate-batch',e.batch||'-'); const riskMap={safe:'资源预估',warning:'资源预估 · 警告',danger:'资源预估 · 风险'}; const model=e.model_size?` · YOLO-${e.model_size}`:''; setText('resource-note',`${riskMap[e.risk]||'资源预估'}${model} · cache=${e.cache_mode||'-'}`); setText('resource-detail',e.note||'估算值仅供参考，实际峰值会随模型、增强策略、驱动和环境波动。')}
async function updateResourceEstimate(){try{const v=collect(); const key=resourceEstimateKey(v); if(key===lastResourceEstimateKey) return; lastResourceEstimateKey=key; setText('resource-note','正在估算...'); const j=await api('/api/train-estimate',{values:v}); showResourceEstimate(j.estimate||{})}catch(e){setText('resource-note','估算失败'); setText('resource-detail',e.message)}}
function scheduleResourceEstimate(){clearTimeout(resourceEstimateTimer); resourceEstimateTimer=setTimeout(updateResourceEstimate,450)}
function apply(v){values={...values,...v}; for(const id of fields){const el=document.getElementById(id); if(el && values[id]!==undefined && el.value!==String(values[id])) el.value=values[id]} updateSplitRatio(); for(const n of ['train_device','train_task','test_source','label_source_type']){if(values[n]!==undefined){const el=document.querySelector(`input[name="${n}"][value="${values[n]}"]`); if(el) el.checked=true}} updateTrainTaskUI(); updateLabelSourceUI(); updateCurrentVideo(); updateCommands(); scheduleResourceEstimate()}


async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); const j=await r.json(); if(!r.ok||j.error) throw new Error(j.error||r.statusText); return j}
let valuesSaveQueue=Promise.resolve();
function saveValues(){const snapshot={...collect()}; valuesSaveQueue=valuesSaveQueue.then(()=>api('/api/values',{values:snapshot})).catch(()=>{}); return valuesSaveQueue}
async function saveDefaults(scope){const snapshot={...collect()}; try{await valuesSaveQueue; const j=await api('/api/defaults',{values:snapshot}); apply(j.values||{}); toast(`${scope||'当前配置'}已保存为默认`)}catch(e){toast(e.message)}}
async function command(action){const j=await api('/api/command',{action,values:collect()}); return j.command}
async function updateCommands(){try{document.getElementById('cmd-train').textContent=await command('train');document.getElementById('cmd-test').textContent=await command('test');document.getElementById('cmd-label').textContent=await command('label')}catch(e){}}
function setInputValue(id,value){const el=document.getElementById(id); if(el){el.value=value; values[id]=value}}
function videoPrefix(video){return video.stem.replace(/[^\w\u4e00-\u9fa5-]+/g,'_').replace(/^_+|_+$/g,'')||'track'}
function videoUrl(video,path='/api/video-file'){return `${path}?path=${encodeURIComponent(video.path)}&t=${Date.now()}`}
function videoSizeText(video){const size=Number(video.size||0); if(!size) return ''; const units=['B','KB','MB','GB','TB']; let n=size,i=0; while(n>=1024&&i<units.length-1){n/=1024;i++} return `${n>=10||i===0?n.toFixed(0):n.toFixed(1)} ${units[i]}`}
function updateVideoPreview(video){const box=document.getElementById('label-video-preview'); const name=document.getElementById('label-preview-name'); if(!box||!name) return; const token=++labelPreviewToken; box.classList.remove('clickable'); box.onclick=null; if(!video){name.textContent='未选择'; box.innerHTML='<span>选择左侧视频后显示首帧预览</span>'; return} name.textContent=video.name; box.classList.add('clickable'); box.innerHTML=`<img src="${videoUrl(video,'/api/video-preview')}" alt="" loading="eager"><div class="play-overlay"><div class="play-button">▶</div></div>`; const img=box.querySelector('img'); if(img){img.onerror=()=>{if(token!==labelPreviewToken) return; box.classList.remove('clickable'); box.onclick=null; box.innerHTML='<span>首帧预览读取失败：可能是视频编码不受当前 OpenCV 支持，但仍可尝试开始标注。</span>'}} box.onclick=()=>playVideoPreview(video)}
function playVideoPreview(video){const box=document.getElementById('label-video-preview'); if(!box) return; ++labelPreviewToken; box.classList.remove('clickable'); box.onclick=null; box.innerHTML=`<video src="${videoUrl(video)}" controls playsinline preload="metadata"></video>`; const player=box.querySelector('video'); if(player){player.onerror=()=>{box.innerHTML='<span>浏览器无法直接播放该视频编码或容器格式；仍可尝试通过网页标注工作台读取并标注。</span>'}; player.play().catch(()=>{})}}
function updateCurrentVideo(){const cur=document.getElementById('label-current-video'); if(!cur) return; const val=(document.getElementById('label_video')?.value||values.label_video||'').trim(); if(labelVideos.length){const matched=labelVideos.findIndex(v=>v.path===val); if(matched!==labelVideoIndex){labelVideoIndex=matched; renderLabelVideos(); updateVideoPreview(labelVideos[labelVideoIndex]||null); return}} cur.textContent=val||'未选择视频'; if(!labelVideos.length) updateVideoPreview(null)}
function renderLabelVideos(){const list=document.getElementById('label-video-list'); const count=document.getElementById('label-video-count'); if(!list||!count) return; const done=labelVideos.filter(v=>v.done).length; const visible=Math.min(labelVisibleCount,labelVideos.length); count.textContent=labelVideos.length?`${done}/${labelVideos.length} 已完成 · 显示 ${visible} 个`:'0 个视频'; list.innerHTML=''; if(!labelVideos.length){list.innerHTML='<div class="empty">当前文件夹没有找到视频。支持 mp4、avi、mov、mkv、wmv、webm 等格式。</div>'; const cur=document.getElementById('label-current-video'); if(cur) cur.textContent=(document.getElementById('label_video')?.value||values.label_video||'').trim()||'未选择视频'; updateVideoPreview(null); return} const fragment=document.createDocumentFragment(); labelVideos.slice(0,visible).forEach((video,idx)=>{const btn=document.createElement('button'); btn.className='video-item'+(idx===labelVideoIndex?' active':'')+(video.done?' done':''); const status=video.done?'已完成':'待标注'; const size=videoSizeText(video); btn.innerHTML=`<b>${idx+1}. ${video.name}</b><span>${status}${size?' · '+size:''} · ${video.rel}</span>`; btn.onclick=()=>selectLabelVideo(idx); fragment.appendChild(btn)}); list.appendChild(fragment); if(visible<labelVideos.length){const more=document.createElement('button'); more.className='video-item'; more.innerHTML=`<b>加载更多视频</b><span>继续显示 ${Math.min(LABEL_VIDEO_PAGE_SIZE,labelVideos.length-visible)} 个，剩余 ${labelVideos.length-visible} 个</span>`; more.onclick=()=>{labelVisibleCount=Math.min(labelVisibleCount+LABEL_VIDEO_PAGE_SIZE,labelVideos.length); renderLabelVideos()}; list.appendChild(more)} const cur=document.getElementById('label-current-video'); const video=labelVideos[labelVideoIndex]; if(cur){cur.textContent=video?video.path:((document.getElementById('label_video')?.value||values.label_video||'').trim()||'未选择视频')}}
function selectLabelVideo(index){if(index<0||index>=labelVideos.length) return; labelVideoIndex=index; if(index>=labelVisibleCount){labelVisibleCount=Math.min(labelVideos.length,Math.ceil((index+1)/LABEL_VIDEO_PAGE_SIZE)*LABEL_VIDEO_PAGE_SIZE)} const video=labelVideos[index]; setInputValue('label_video',video.path); if(!rawLabelPrefix.manual){setInputValue('label_prefix',videoPrefix(video))} renderLabelVideos(); updateVideoPreview(video); saveValues(); updateCommands()}
async function loadLabelVideos(){try{await saveValues(); const list=document.getElementById('label-video-list'); if(list) list.innerHTML='<div class="empty">正在读取视频文件夹，请稍候...</div>'; const r=await fetch('/api/label-videos'); const j=await r.json(); if(j.error) throw new Error(j.error); labelVideos=j.items||[]; labelVisibleCount=LABEL_VIDEO_PAGE_SIZE; const current=(document.getElementById('label_video')?.value||'').trim(); labelVideoIndex=labelVideos.findIndex(v=>v.path===current); if(labelVideoIndex<0&&labelVideos.length) labelVideoIndex=0; if(labelVideos.length) selectLabelVideo(labelVideoIndex); else renderLabelVideos(); toast(`已读取 ${labelVideos.length} 个视频${labelVideos.length>=2000?'，已自动限制前 2000 个':''}`)}catch(e){toast(e.message)}}

async function pickTestImage(){try{const j=await api('/api/pick-test-image',{values:collect()}); if(j.path){setInputValue('test_image_file',j.path); await saveValues(); updateCommands()}else{toast('未选择图片')}}catch(e){toast(e.message)}}
async function pickTestImageFolder(){try{const j=await api('/api/pick-test-image-folder',{values:collect()}); if(j.path){setInputValue('test_image_folder',j.path); await saveValues(); updateCommands()}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
async function pickTestOutputDir(){try{const j=await api('/api/pick-test-output-dir',{values:collect()}); if(j.path){setInputValue('test_output_dir',j.path); await saveValues(); updateCommands()}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
async function pickLabelVideoDir(){try{const j=await api('/api/pick-label-video-dir',{values:collect()}); if(j.path){setInputValue('label_video_dir',j.path); await saveValues(); await loadLabelVideos()}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
async function pickLabelImagesDir(){try{const j=await api('/api/pick-label-images-dir',{values:collect()}); if(j.path){setInputValue('label_images_input_dir',j.path); await saveValues(); toast('已选择图片集文件夹')}else{toast('未选择文件夹')}}catch(e){toast(e.message)}}
function selectNextVideo(){if(!labelVideos.length){toast('请先读取文件夹视频'); return} if(labelVideoIndex>=0&&labelVideos[labelVideoIndex]) labelVideos[labelVideoIndex].done=true; const next=Math.min(labelVideos.length-1,labelVideoIndex+1); selectLabelVideo(next); toast(next===labelVideos.length-1?'已到最后一个视频':'已切换到下一个视频')}

function selectPrevVideo(){if(!labelVideos.length){toast('请先读取文件夹视频'); return} selectLabelVideo(Math.max(0,labelVideoIndex-1))}
let labelSessionId=sessionStorage.getItem('labelSessionId')||'';
let labelSessionState=null;
let labelPlaying=false;
let labelAdvanceBusy=false;
let labelPlayTimer=null;
let labelActiveObjectId=null;
const labelHiddenObjectIds=new Set();
let labelDrawMode='add';
let labelDragStart=null;
let labelChoiceResolver=null;
function labelFrameUrl(){return `/api/label-session/frame?session_id=${encodeURIComponent(labelSessionId)}&t=${Date.now()}`}
function labelPalette(i){return ['#50dc78','#50b4ff','#e678ff','#ffbe46','#78dcff','#b4a0ff','#78ffd2','#ff8c8c'][i%8]}
function showLabelStudio(show){const studio=document.getElementById('label-browser-studio'); if(studio) studio.hidden=!show}
function eyeIcon(hidden){return hidden?'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 3 18 18"></path><path d="M10.6 6.2A10.8 10.8 0 0 1 12 6c5.5 0 9.3 6 9.3 6a17.8 17.8 0 0 1-3.1 3.8M6.2 6.2A17.6 17.6 0 0 0 2.7 12S6.5 18 12 18c.8 0 1.6-.1 2.4-.4"></path><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"></path></svg>':'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.7 12S6.5 6 12 6s9.3 6 9.3 6-3.8 6-9.3 6S2.7 12 2.7 12Z"></path><circle cx="12" cy="12" r="3"></circle></svg>'}
function renderLabelSession(){const state=labelSessionState; const status=document.getElementById('label-session-status'); const list=document.getElementById('label-object-list'); const tip=document.getElementById('label-session-tip'); const seek=document.getElementById('label-frame-seek'); const seekValue=document.getElementById('label-frame-seek-value'); if(!state||!status||!list) return; if(seek){seek.disabled=state.source_type!=='video'||!state.frame_count; seek.max=String(Math.max(0,(state.frame_count||1)-1)); seek.value=String(state.frame_index||0)} if(seekValue) seekValue.textContent=`帧 ${state.frame_index}${state.frame_count?` / ${Math.max(0,state.frame_count-1)}`:''}`; status.innerHTML=`<span>来源：${state.source_type==='camera'?'摄像头':state.source_type==='images'?'图片集':'视频'}</span><span>帧：${state.frame_index}${state.frame_count?'/'+Math.max(0,state.frame_count-1):''}</span><span>已保存：${state.saved}</span><span>间隔：${state.interval}</span><span>目标：${state.objects.filter(x=>x.ok&&!x.hidden).length}/${state.objects.filter(x=>!x.hidden).length}</span>${state.ended?'<span>来源已结束</span>':''}${state.lost?'<span>跟踪丢失，请修正或删除</span>':''}`; if(tip) tip.textContent=state.lost?'跟踪丢失，已暂停播放；请选择目标后修正或删除。':'在画面上拖动鼠标绘制标注框。'; list.innerHTML=''; if(!state.objects.length){list.innerHTML='<div class="empty">尚未框选目标。点击“添加框”后在画面拖动鼠标。</div>'} else {state.objects.forEach((obj,index)=>{const hidden=Boolean(obj.hidden)||labelHiddenObjectIds.has(obj.id); const item=document.createElement('div'); item.className='label-object'+(obj.id===labelActiveObjectId?' active':'')+(!obj.ok?' lost':''); item.innerHTML=`<button class="label-object-visibility${hidden?' is-hidden':''}" type="button" title="${hidden?'显示':'隐藏'} #${obj.id} ${obj.label} 的框" aria-label="${hidden?'显示':'隐藏'} #${obj.id} ${obj.label} 的框">${eyeIcon(hidden)}</button><div class="label-object-info"><b>#${obj.id} ${obj.label}${obj.ok?'':' · 已丢失'}</b><span>${obj.w} × ${obj.h} · 参考视角 ${obj.sample_count||1}</span></div>`; item.onclick=()=>{labelActiveObjectId=obj.id; renderLabelSession(); drawLabelCanvas()}; item.querySelector('.label-object-visibility').onclick=async event=>{event.stopPropagation(); const nextHidden=!hidden; try{const j=await api('/api/label-session/object',{session_id:labelSessionId,action:'visibility',object_id:obj.id,hidden:nextHidden}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; if(nextHidden) labelHiddenObjectIds.add(obj.id); else labelHiddenObjectIds.delete(obj.id); renderLabelSession()}catch(e){toast(e.message)}}; list.appendChild(item)})} const button=document.getElementById('label-play-button'); if(button) button.textContent=labelPlaying?'暂停跟踪':'播放跟踪'; drawLabelCanvas()}

function refreshLabelFrame(){const image=document.getElementById('label-frame-image'); const stage=document.getElementById('label-stage'); if(!image||!labelSessionId) return; image.hidden=false; stage?.classList.remove('empty-stage'); image.onload=()=>drawLabelCanvas(); image.src=labelFrameUrl()}
function setupLabelTimeline(){const seek=document.getElementById('label-frame-seek'); const value=document.getElementById('label-frame-seek-value'); if(!seek||seek.dataset.ready) return; seek.dataset.ready='1'; seek.addEventListener('input',()=>{if(value) value.textContent=`帧 ${seek.value}${labelSessionState?.frame_count?` / ${Math.max(0,labelSessionState.frame_count-1)}`:''}`}); seek.addEventListener('change',()=>seekBrowserLabelFrame(Number(seek.value)))}
function setupLabelCanvas(){const canvas=document.getElementById('label-frame-canvas'); if(!canvas||canvas.dataset.ready) return; canvas.dataset.ready='1'; canvas.hidden=false; canvas.addEventListener('pointerdown',event=>{if(!labelSessionState) return; const point=labelCanvasPoint(event); if(!point) return; labelDragStart=point; canvas.setPointerCapture(event.pointerId); drawLabelCanvas(point)}); canvas.addEventListener('pointermove',event=>{if(!labelDragStart) return; drawLabelCanvas(labelCanvasPoint(event))}); canvas.addEventListener('pointerup',async event=>{if(!labelDragStart) return; const end=labelCanvasPoint(event); const start=labelDragStart; labelDragStart=null; canvas.releasePointerCapture(event.pointerId); drawLabelCanvas(); if(!end) return; const x=Math.min(start.x,end.x),y=Math.min(start.y,end.y),w=Math.abs(end.x-start.x),h=Math.abs(end.y-start.y); if(w<5||h<5){toast('标注框过小');return} const action=labelDrawMode==='edit'?'update':labelDrawMode==='sample'?'add_sample':'add'; if(action==='add_sample'&&!labelActiveObjectId){toast('请先在右侧选择要追加视角的目标');return} let label=''; if(action==='add'){label=await chooseBrowserLabel(); if(!label) return} try{const j=await api('/api/label-session/object',{session_id:labelSessionId,action,object_id:labelActiveObjectId,bbox:{x,y,w,h},label}); labelSessionState=j.state; if(action==='add') labelActiveObjectId=labelSessionState.objects.at(-1)?.id||null; renderLabelSession()}catch(e){toast(e.message)}}); window.addEventListener('resize',()=>drawLabelCanvas())}

function labelCanvasPoint(event){const image=document.getElementById('label-frame-image'); if(!image||!labelSessionState) return null; const rect=image.getBoundingClientRect(); if(!rect.width||!rect.height) return null; return {x:(event.clientX-rect.left)*labelSessionState.width/rect.width,y:(event.clientY-rect.top)*labelSessionState.height/rect.height}}
function drawLabelCanvas(dragEnd=null){const canvas=document.getElementById('label-frame-canvas'); const image=document.getElementById('label-frame-image'); if(!canvas||!image||!labelSessionState||image.hidden) return; const rect=image.getBoundingClientRect(); const dpr=window.devicePixelRatio||1; canvas.style.width=rect.width+'px'; canvas.style.height=rect.height+'px'; canvas.style.left=(rect.left-canvas.parentElement.getBoundingClientRect().left)+'px'; canvas.style.top=(rect.top-canvas.parentElement.getBoundingClientRect().top)+'px'; canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr)); const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,rect.width,rect.height); const sx=rect.width/labelSessionState.width,sy=rect.height/labelSessionState.height; labelSessionState.objects.forEach((obj,index)=>{if(obj.hidden||labelHiddenObjectIds.has(obj.id)) return; const active=obj.id===labelActiveObjectId; ctx.strokeStyle=obj.ok?labelPalette(index):'#ff6678'; ctx.lineWidth=active?3:2; ctx.strokeRect(obj.x*sx,obj.y*sy,obj.w*sx,obj.h*sy); ctx.font='600 13px Microsoft YaHei UI'; const text=`#${obj.id} ${obj.label}${obj.ok?'':' LOST'}`; const tx=obj.x*sx,ty=Math.max(18,obj.y*sy-5); ctx.fillStyle='rgba(3,8,19,.85)'; const tw=ctx.measureText(text).width+10; ctx.fillRect(tx,ty-16,tw,20); ctx.fillStyle='#fff'; ctx.fillText(text,tx+5,ty)}); if(labelDragStart&&dragEnd){const x=Math.min(labelDragStart.x,dragEnd.x)*sx,y=Math.min(labelDragStart.y,dragEnd.y)*sy,w=Math.abs(labelDragStart.x-dragEnd.x)*sx,h=Math.abs(labelDragStart.y-dragEnd.y)*sy; ctx.strokeStyle='#fff';ctx.setLineDash([6,4]);ctx.lineWidth=2;ctx.strokeRect(x,y,w,h);ctx.setLineDash([])}}
function closeLabelChoice(value=''){const dialog=document.getElementById('label-choice-dialog'); if(dialog) dialog.hidden=true; document.body.style.overflow=''; const resolve=labelChoiceResolver; labelChoiceResolver=null; if(resolve) resolve(value)}
function chooseBrowserLabel(){const labels=(labelSessionState?.labels||[]); const choices=labels.length?labels:(collect().label_name||'object').split(/[,;\n]+/).map(x=>x.trim()).filter(Boolean); if(choices.length===1) return Promise.resolve(choices[0]); return new Promise(resolve=>{const dialog=document.getElementById('label-choice-dialog'); const options=document.getElementById('label-choice-options'); const cancel=document.getElementById('label-choice-cancel'); if(!dialog||!options){resolve('');return} labelChoiceResolver=resolve; options.innerHTML=''; choices.forEach((label,index)=>{const button=document.createElement('button'); button.type='button'; button.className='label-choice-option'; button.innerHTML=`<span class="label-choice-option-key">${index<9?index+1:''}</span>`; const text=document.createElement('span'); text.textContent=label; button.appendChild(text); button.onclick=()=>closeLabelChoice(label); options.appendChild(button)}); cancel.onclick=()=>closeLabelChoice(''); dialog.onclick=event=>{if(event.target===dialog) closeLabelChoice('')}; dialog.onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();closeLabelChoice('');return} const index=Number(event.key)-1; if(Number.isInteger(index)&&index>=0&&index<choices.length){event.preventDefault();closeLabelChoice(choices[index])}}; dialog.hidden=false; document.body.style.overflow='hidden'; requestAnimationFrame(()=>options.querySelector('button')?.focus())})}
function setLabelDrawMode(mode){if(!labelSessionId){toast('请先开始网页标注');return} if((mode==='edit'||mode==='sample')&&!labelActiveObjectId){toast('请先在右侧选择目标');return} if(mode==='sample'&&labelSessionState?.tracker!=='multi_template'){toast('追加视角需要先选择 Multi-template（多角度）跟踪器');return} labelDrawMode=mode; toast(mode==='edit'?'请拖动绘制选中目标的新框':mode==='sample'?'请拖动绘制该目标在新角度下的框':'请拖动绘制新目标框')}

async function startBrowserLabelSession(){const v=collect(); if(v.label_source_type==='images'&&!v.label_images_input_dir.trim()){toast('请填写图片集文件夹路径');return} if(v.label_source_type==='camera'&&!/^\d+$/.test(v.label_camera_index.trim())){toast('请输入非负整数摄像头索引，例如 0');return} if(v.label_source_type==='video'&&!v.label_video.trim()){toast('请先从队列选择视频或填写视频路径');return} if(labelSessionId) await endBrowserLabelSession(); try{const j=await api('/api/label-session/start',{values:v}); labelSessionId=j.state.session_id; sessionStorage.setItem('labelSessionId',labelSessionId); labelSessionState={...j.state,labels:(v.label_name||'object').split(/[,;\n]+/).map(x=>x.trim()).filter(Boolean)}; labelActiveObjectId=null; labelHiddenObjectIds.clear(); labelPlaying=false; showLabelStudio(true); setupLabelTimeline(); setupLabelCanvas(); renderLabelSession(); refreshLabelFrame(); toast('网页标注已就绪，请添加目标框')}catch(e){toast(e.message)}}
async function advanceBrowserLabelFrame(){if(!labelSessionId||labelAdvanceBusy) return; labelAdvanceBusy=true; try{const j=await api('/api/label-session/advance',{session_id:labelSessionId}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; if(labelSessionState.lost||labelSessionState.ended) stopBrowserLabelPlay(); renderLabelSession(); refreshLabelFrame()}catch(e){stopBrowserLabelPlay();toast(e.message)}finally{labelAdvanceBusy=false}}
async function seekBrowserLabelFrame(frameIndex){if(!labelSessionId||labelAdvanceBusy||labelSessionState?.source_type!=='video') return; labelAdvanceBusy=true; stopBrowserLabelPlay(); try{const j=await api('/api/label-session/seek',{session_id:labelSessionId,frame_index:frameIndex}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; labelActiveObjectId=labelSessionState.objects[0]?.id||null; renderLabelSession(); refreshLabelFrame(); toast('已跳转到指定帧，请修正目标框后继续')}catch(e){toast(e.message)}finally{labelAdvanceBusy=false}}
function toggleBrowserLabelPlay(){if(labelPlaying) stopBrowserLabelPlay();else startBrowserLabelPlay()}
function startBrowserLabelPlay(){if(!labelSessionId){toast('请先开始网页标注');return} if(!labelSessionState?.objects.length){toast('请先添加至少一个目标框');return} if(labelSessionState.lost){toast('请先修正或删除丢失目标');return} labelPlaying=true; renderLabelSession(); const tick=async()=>{if(!labelPlaying) return; await advanceBrowserLabelFrame(); if(labelPlaying) labelPlayTimer=setTimeout(tick,45)}; tick()}
function stopBrowserLabelPlay(){labelPlaying=false;clearTimeout(labelPlayTimer);labelPlayTimer=null;renderLabelSession()}
async function saveBrowserLabelFrame(){if(!labelSessionId) return; try{const j=await api('/api/label-session/save',{session_id:labelSessionId}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; renderLabelSession(); if(j.saved){toast('当前帧已保存');loadLabelResults()}else toast('没有可保存的有效目标框')}catch(e){toast(e.message)}}
async function deleteBrowserLabelObject(){if(!labelSessionId||!labelActiveObjectId){toast('请先选择目标');return} try{const j=await api('/api/label-session/object',{session_id:labelSessionId,action:'delete',object_id:labelActiveObjectId,bbox:{x:0,y:0,w:3,h:3}}); labelSessionState={...j.state,labels:labelSessionState?.labels||[]}; labelActiveObjectId=labelSessionState.objects[0]?.id||null; renderLabelSession()}catch(e){toast(e.message)}}
async function endBrowserLabelSession(){stopBrowserLabelPlay(); if(!labelSessionId){showLabelStudio(false);return} try{await api('/api/label-session/end',{session_id:labelSessionId})}catch(e){} labelSessionId='';labelSessionState=null;labelActiveObjectId=null;labelHiddenObjectIds.clear();sessionStorage.removeItem('labelSessionId');showLabelStudio(false);loadLabelResults()}
async function runLabelCurrent(){await startBrowserLabelSession()}
async function copyCommand(action){try{const cmd=await command(action); await navigator.clipboard.writeText(cmd); toast('命令已复制')}catch(e){toast(e.message)}}
async function runAction(action){try{await api('/api/run',{action,values:collect()}); showTab('logs'); toast('任务已启动'); refreshState()}catch(e){toast(e.message)}}
function updateJobInputMode(){const input=document.getElementById('job-input'); const secret=document.getElementById('job-input-secret'); if(input&&secret) input.type=secret.checked?'password':'text'}
async function stopJob(){try{const j=await api('/api/stop',{}); toast(j.stopped?'已请求停止':'当前没有正在运行的任务'); refreshState()}catch(e){toast(e.message)}}
function openImageLightbox(src,title){const box=document.getElementById('image-lightbox'); const image=document.getElementById('image-lightbox-image'); const caption=document.getElementById('image-lightbox-title'); if(!box||!image||!caption) return; image.src=src; image.alt=title; caption.textContent=title; box.hidden=false; document.body.style.overflow='hidden'; document.getElementById('image-lightbox-close')?.focus()}
function closeImageLightbox(){const box=document.getElementById('image-lightbox'); if(!box||box.hidden) return; box.hidden=true; document.body.style.overflow=''}
async function loadLabelResults(){try{await saveValues(); const r=await fetch('/api/label-results'); const j=await r.json(); const box=document.getElementById('label-results'); box.innerHTML=''; const items=j.items||[]; if(!items.length){box.innerHTML='<div class="empty">当前图片目录和标注目录中还没有可显示的标注结果。</div>'; return} for(const it of items){const card=document.createElement('div'); card.className='sample'; const src='/api/label-preview?image='+encodeURIComponent(it.image)+'&xml='+encodeURIComponent(it.xml)+'&t='+Date.now(); const title=`${it.stem} · ${(it.boxes||[]).length} 个框`; card.innerHTML=`<button class="preview-trigger" type="button" aria-label="放大查看 ${title}"><img src="${src}" loading="lazy" alt="${title}"></button><div class="meta"><b>${it.stem}</b><span>${(it.boxes||[]).length} 个框</span><button class="delete">删除标注</button></div>`; card.querySelector('.preview-trigger').onclick=()=>openImageLightbox(src,title); card.querySelector('.delete').onclick=async()=>{const imageMode=collect().label_source_type==='images'; const target=imageMode?'对应 XML 标注':'这张图片和对应 XML'; if(Date.now()>deleteConfirmUntil){if(!confirm(`确定删除${target}吗？\n确认后 5 分钟内删除标注不再重复询问。`)) return; deleteConfirmUntil=Date.now()+5*60*1000} await api('/api/delete-label-sample',{image:it.image,xml:it.xml}); card.remove(); toast(imageMode?'已删除 XML 标注':'已删除废图和 XML')}; box.appendChild(card)}}catch(e){toast(e.message)}}
async function loadTrainPlots(){try{const r=await fetch('/api/train-plots'); const j=await r.json(); const box=document.getElementById('train-plots'); if(!box) return; const note=document.getElementById('train-plots-note'); const items=j.items||[]; if(note) note.textContent=items.length?`已发现 ${items.length} 张图片`:'训练开始后自动刷新'; if(!items.length){box.innerHTML='<div class="empty">训练进行中或尚未生成可视化图。</div>'; return} box.innerHTML=''; for(const item of items){const card=document.createElement('div'); card.className='sample'; card.innerHTML=`<img src="/api/train-plot?name=${encodeURIComponent(item.name)}&t=${Date.now()}" loading="lazy"><div class="meta"><b>${item.name}</b></div>`; box.appendChild(card)}}catch(e){}}
function scheduleLabelResultsRefresh(){clearTimeout(labelResultsTimer); labelResultsTimer=setTimeout(()=>loadLabelResults(),350)}
async function refreshState(){const r=await fetch('/api/state'); const s=await r.json(); apply(s.values||{}); updateTrainProgress(s.train_progress||{}); const log=document.getElementById('log'); log.textContent=(s.logs||[]).join(''); log.scrollTop=log.scrollHeight; const pill=document.getElementById('runPill'); pill.className='pill '+(s.running?'run':'idle'); pill.querySelector('span:last-child').textContent=s.running?'运行中':'空闲'; document.getElementById('jobInfo').textContent=s.job?`${s.job} | 开始: ${s.started_at||'-'} | 结束: ${s.finished_at||'-'} | 退出码: ${s.exit_code??'-'}`:'暂无任务'; const box=document.getElementById('markers'); box.innerHTML=''; for(const [k,v] of Object.entries(s.markers||{})){const div=document.createElement('div'); div.className='marker'; div.innerHTML=`<b>${k}</b><span>${v}</span>`; box.appendChild(div)}}
function copyLogs(){navigator.clipboard.writeText(document.getElementById('log').textContent);toast('日志已复制')}
function showTab(name){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+name).classList.add('active');document.querySelectorAll('.nav button').forEach(x=>x.classList.toggle('active',x.dataset.tab===name)); if(name==='label') loadLabelResults()}

function installLabelCanvasInputGuard(){const canvas=document.getElementById('label-frame-canvas'); const image=document.getElementById('label-frame-image'); if(canvas&&!canvas.dataset.inputGuard){canvas.dataset.inputGuard='1'; const prevent=event=>event.preventDefault(); canvas.addEventListener('pointerdown',prevent,{capture:true}); canvas.addEventListener('pointermove',prevent,{capture:true}); canvas.addEventListener('pointerup',prevent,{capture:true}); canvas.addEventListener('pointercancel',event=>{event.preventDefault(); labelDragStart=null; drawLabelCanvas()},{capture:true}); canvas.addEventListener('lostpointercapture',()=>{labelDragStart=null; drawLabelCanvas()}); canvas.addEventListener('dragstart',prevent)} if(image) image.addEventListener('dragstart',event=>event.preventDefault())}
installLabelCanvasInputGuard();
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
document.getElementById('image-lightbox-close')?.addEventListener('click',closeImageLightbox);
document.getElementById('image-lightbox')?.addEventListener('click',event=>{if(event.target===event.currentTarget) closeImageLightbox()});
document.addEventListener('keydown',event=>{if(event.key==='Escape') closeImageLightbox()});
document.querySelectorAll('input,select').forEach(el=>{const handler=()=>{if(el.id==='label_prefix') rawLabelPrefix.manual=true; if(el.name==='train_task') updateTrainTaskUI(); if(el.name==='label_source_type') updateLabelSourceUI(); collect(); updateCurrentVideo(); saveValues(); updateCommands(); if(['train_images_dir','base_model','img_width','img_height','image_resize_mode','batch','train_cache'].includes(el.id)||el.name==='train_device'||el.name==='train_task') scheduleResourceEstimate(); if(['label_images_dir','label_annotations_dir','label_annotations_dir_images','label_images_input_dir'].includes(el.id)) scheduleLabelResultsRefresh()}; el.addEventListener('input',handler); el.addEventListener('change',handler)});
updateSplitRatio();
refreshState(); setInterval(refreshState,1400);



</script>
</body>
</html>'''


class PanelHandler(BaseHTTPRequestHandler):
    server_version = "YOLOWebPanel/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return


    def send_json(self, data: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/state":
            with STATE_LOCK:
                self.send_json({
                    "values": STATE["values"],
                    "logs": STATE["logs"],
                    "markers": STATE["markers"],
                    "train_progress": STATE["train_progress"],
                    "running": STATE["running"],
                    "job": STATE["job"],
                    "exit_code": STATE["exit_code"],
                    "started_at": STATE["started_at"],
                    "finished_at": STATE["finished_at"],
                    "last_error": STATE["last_error"],
                })
            return
        if parsed.path == "/api/train-plots":
            _, items = list_train_plots()
            self.send_json({"items": items})
            return
        if parsed.path == "/api/train-plot":
            try:
                params = parse_qs(parsed.query)
                name = params.get("name", [""])[0]
                plot_dir, _ = list_train_plots()
                if not plot_dir.is_dir():
                    raise ValueError("当前没有可用的训练图片目录。")
                send_train_plot(self, plot_dir, name)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/label-results":
            with STATE_LOCK:
                values = STATE["values"].copy()
            self.send_json({"items": list_label_results(values)})
            return
        if parsed.path == "/api/label-videos":
            with STATE_LOCK:
                values = STATE["values"].copy()
            self.send_json({"items": list_label_videos(values)})
            return
        if parsed.path == "/api/video-preview":
            try:
                params = parse_qs(parsed.query)
                video = params.get("path", [""])[0]
                with STATE_LOCK:
                    values = STATE["values"].copy()
                video_dir = Path(values.get("label_video_dir", "")).expanduser().resolve()
                if not video_dir.is_dir():
                    raise ValueError("请先选择有效的视频文件夹。")
                video_path = resolve_under(video, video_dir)
                raw = render_video_preview(video_path)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/video-file":
            try:
                params = parse_qs(parsed.query)
                video = params.get("path", [""])[0]
                with STATE_LOCK:
                    values = STATE["values"].copy()
                video_dir = Path(values.get("label_video_dir", "")).expanduser().resolve()
                if not video_dir.is_dir():
                    raise ValueError("请先选择有效的视频文件夹。")
                video_path = resolve_under(video, video_dir)
                send_video_file(self, video_path)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/label-session/frame":
            try:
                params = parse_qs(parsed.query)
                session = get_label_session(params.get("session_id", [""])[0])
                with session["lock"]:
                    ok, encoded = cv2.imencode(".jpg", session["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if not ok:
                        raise ValueError("当前标注帧编码失败。")
                    raw = encoded.tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        if parsed.path == "/api/label-preview":
            try:
                params = parse_qs(parsed.query)
                image = params.get("image", [""])[0]
                xml = params.get("xml", [""])[0]
                with STATE_LOCK:
                    values = STATE["values"].copy()
                image_path = resolve_under(image, label_result_images_dir(values))
                xml_path = resolve_under(xml, Path(values["label_annotations_dir"]))
                raw = render_label_preview(image_path, xml_path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                append_log(f"[网页标注] 标注结果预览失败：{exc}\n")
                message = str(exc).encode("utf-8", errors="replace")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return
        self.send_error(404)



    def do_POST(self) -> None:
        try:
            body = self.read_json()
            if self.path == "/api/command":
                action = str(body.get("action", ""))
                values = clean_values(body.get("values"))
                cmd = command_for(action, values)
                self.send_json({"command": quote_cmd(cmd)})
                return
            if self.path == "/api/train-estimate":
                values = clean_values(body.get("values"))
                self.send_json({"estimate": estimate_train_resources(values)})
                return
            if self.path == "/api/values":

                values = clean_values(body.get("values"))
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True})
                return
            if self.path == "/api/defaults":
                values = save_user_defaults(body.get("values") or {})
                with STATE_LOCK:
                    STATE["values"] = values.copy()
                self.send_json({"ok": True, "values": values, "path": str(USER_DEFAULTS_FILE)})
                return
            if self.path == "/api/pick-test-image":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_image_file(values.get("test_image_file", ""))})
                return
            if self.path == "/api/pick-test-image-folder":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_directory(values.get("test_image_folder", ""))})
                return
            if self.path == "/api/pick-test-output-dir":
                values = clean_values(body.get("values"))
                self.send_json({"path": pick_directory(values.get("test_output_dir", ""))})
                return
            if self.path == "/api/pick-label-video-dir":
                values = clean_values(body.get("values"))
                selected = pick_directory(values.get("label_video_dir", ""))
                if selected:
                    values["label_video_dir"] = selected
                    with STATE_LOCK:
                        STATE["values"] = values.copy()
                self.send_json({"path": selected})
                return
            if self.path == "/api/pick-label-images-dir":
                values = clean_values(body.get("values"))
                selected = pick_directory(values.get("label_images_input_dir", ""))
                if selected:
                    values["label_images_input_dir"] = selected
                    with STATE_LOCK:
                        STATE["values"] = values.copy()
                self.send_json({"path": selected})
                return
            if self.path == "/api/label-session/start":
                values = clean_values(body.get("values"))
                state = start_label_session(values)
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/advance":
                session = get_label_session(str(body.get("session_id", "")))
                with session["lock"]:
                    state = advance_label_session(session)
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/seek":
                session = get_label_session(str(body.get("session_id", "")))
                with session["lock"]:
                    state = seek_label_session(session, int(body.get("frame_index", 0)))
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/object":
                session = get_label_session(str(body.get("session_id", "")))
                action = str(body.get("action", "add"))
                bbox_raw = body.get("bbox") or {}
                with session["lock"]:
                    frame = session["frame"]
                    bbox = sanitize_label_bbox(
                        (bbox_raw.get("x", 0), bbox_raw.get("y", 0), bbox_raw.get("w", 0), bbox_raw.get("h", 0)),
                        frame.shape[1], frame.shape[0],
                    )
                    if action == "visibility":
                        object_id = int(body.get("object_id", 0))
                        obj = next((item for item in session["objects"] if item.obj_id == object_id), None)
                        if obj is None:
                            raise ValueError("要切换显示状态的目标不存在。")
                        obj.hidden = bool(body.get("hidden", False))
                    elif action == "delete":
                        object_id = int(body.get("object_id", 0))
                        session["objects"] = [obj for obj in session["objects"] if obj.obj_id != object_id]
                    else:
                        if bbox[2] < 3 or bbox[3] < 3:
                            raise ValueError("标注框过小，请重新绘制。")
                        if action in {"update", "add_sample"}:
                            object_id = int(body.get("object_id", 0))
                            obj = next((item for item in session["objects"] if item.obj_id == object_id), None)
                            if obj is None:
                                raise ValueError("要修正的目标不存在。")
                            if action == "add_sample":
                                add_sample = getattr(obj.tracker, "add_sample", None)
                                if not callable(add_sample):
                                    raise ValueError("追加视角仅支持 Multi-template tracker，请重新开始会话后选择该模式。")
                                if not add_sample(frame, bbox):
                                    raise ValueError("参考视角采集失败，请重新绘制更大的框。")
                                obj.bbox = bbox
                                obj.ok = True
                                obj.sample_count += 1
                            else:
                                obj.bbox = bbox
                                obj.tracker = make_label_tracker(session["tracker"])
                                obj.ok = init_label_tracker(obj.tracker, frame, bbox)
                                obj.sample_count = 1
                        else:
                            label = str(body.get("label", "")).strip()
                            if label not in session["labels"]:
                                raise ValueError("请选择当前标签列表中的类别。")
                            tracker = make_label_tracker(session["tracker"])
                            if not init_label_tracker(tracker, frame, bbox):
                                raise ValueError("跟踪器初始化失败，请换一个更大的标注框。")
                            session["objects"].append(LabelTrackObject(session["next_object_id"], label, bbox, tracker))
                            session["next_object_id"] += 1
                    state = label_session_state(session)
                self.send_json({"ok": True, "state": state})
                return
            if self.path == "/api/label-session/save":
                session = get_label_session(str(body.get("session_id", "")))
                with session["lock"]:
                    saved = save_label_session_sample(session)
                    state = label_session_state(session)
                self.send_json({"ok": True, "saved": saved, "state": state})
                return
            if self.path == "/api/label-session/end":
                end_label_session(str(body.get("session_id", "")))
                self.send_json({"ok": True})
                return



            if self.path == "/api/run":
                action = str(body.get("action", ""))
                values = clean_values(body.get("values"))
                start_job(action, values)
                self.send_json({"ok": True})
                return
            if self.path == "/api/stop":
                self.send_json({"stopped": stop_job()})
                return

            if self.path == "/api/delete-label-sample":
                image = str(body.get("image", ""))
                xml = str(body.get("xml", ""))
                with STATE_LOCK:
                    values = STATE["values"].copy()
                image_path = resolve_under(image, label_result_images_dir(values))
                xml_path = resolve_under(xml, Path(values["label_annotations_dir"]))
                paths = (xml_path,) if values.get("label_source_type") == "images" else (image_path, xml_path)
                deleted = []
                for path in paths:
                    if path.exists() and path.is_file():
                        path.unlink()
                        deleted.append(str(path))
                self.send_json({"ok": True, "deleted": deleted})
                return
            self.send_error(404)

        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)


def main() -> None:
    parser = argparse.ArgumentParser(description="model-training-tool Linux Web Panel")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8989)


    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    with STATE_LOCK:
        STATE["values"] = load_user_defaults()

    url = f"http://127.0.0.1:{args.port}"
    try:
        server = ThreadingHTTPServer((args.host, args.port), PanelHandler)
    except OSError as exc:
        print(f"无法监听 {args.host}:{args.port}：{exc}", file=sys.stderr)
        print("请确认 8989 端口未被占用，并允许 Python 通过防火墙。", file=sys.stderr)


        raise SystemExit(1) from exc
    print(f"model-training-tool: {url}")
    print(f"Listening on {args.host}:{args.port}")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_job()
        server.server_close()


if __name__ == "__main__":
    main()
