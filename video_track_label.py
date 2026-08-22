import argparse
import re
import sys
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2


BBox = Tuple[int, int, int, int]


@dataclass
class TrackObject:
    obj_id: int
    label: str
    bbox: BBox
    tracker: object
    ok: bool = True
    sample_count: int = 1


class TemplateTracker:
    def __init__(self, search_scale: float = 2.5, min_score: float = 0.45):
        self.search_scale = search_scale
        self.min_score = min_score
        self.template = None
        self.bbox: Optional[BBox] = None

    def init(self, frame, bbox: BBox) -> bool:
        x, y, w, h = sanitize_bbox(bbox, frame.shape[1], frame.shape[0])
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
        pad_x = max(int(w * self.search_scale), 20)
        pad_y = max(int(h * self.search_scale), 20)
        sx1 = max(0, x - pad_x)
        sy1 = max(0, y - pad_y)
        sx2 = min(frame.shape[1], x + w + pad_x)
        sy2 = min(frame.shape[0], y + h + pad_y)
        search = gray[sy1:sy2, sx1:sx2]
        if search.shape[0] < h or search.shape[1] < w:
            return False, self.bbox
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        nx, ny = sx1 + loc[0], sy1 + loc[1]
        self.bbox = sanitize_bbox((nx, ny, w, h), frame.shape[1], frame.shape[0])
        return score >= self.min_score, self.bbox


class MultiTemplateTracker:
    """以 CSRT 为主跟踪，多个参考模板仅用于 CSRT 丢失后的恢复。"""
    def __init__(self, search_scale: float = 2.5, min_score: float = 0.55):
        self.search_scale = search_scale
        self.min_score = min_score
        self.samples: list[tuple[object, int, int]] = []
        self.bbox: Optional[BBox] = None
        self.primary = None

    def _reset_primary(self, frame, bbox: BBox) -> None:
        self.primary = make_cv_tracker("csrt")
        if self.primary is not None and not init_tracker(self.primary, frame, bbox):
            self.primary = None

    def init(self, frame, bbox: BBox) -> bool:
        if make_cv_tracker("csrt") is None:
            raise RuntimeError(
                "multi_template requires OpenCV CSRT. Install opencv-contrib-python, then restart the labeling tool."
            )
        self.samples = []
        self.bbox = None
        self.primary = None
        return self.add_sample(frame, bbox)

    def add_sample(self, frame, bbox: BBox) -> bool:
        x, y, w, h = sanitize_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 2 or h <= 2:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.samples.append((gray[y:y + h, x:x + w].copy(), w, h))
        self.bbox = (x, y, w, h)
        self._reset_primary(frame, self.bbox)
        return True

    def _recover_from_templates(self, frame) -> tuple[bool, BBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        pad_x = max(int(w * self.search_scale), 20)
        pad_y = max(int(h * self.search_scale), 20)
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
                best_bbox = sanitize_bbox((sx1 + loc[0], sy1 + loc[1], sample_w, sample_h), frame.shape[1], frame.shape[0])
        if best_score < self.min_score:
            return False, self.bbox
        self.bbox = best_bbox
        self._reset_primary(frame, self.bbox)
        return True, self.bbox

    def update(self, frame):
        if not self.samples or self.bbox is None:
            return False, (0, 0, 0, 0)
        if self.primary is not None:
            ok, bbox = update_tracker(self.primary, frame)
            if ok:
                self.bbox = sanitize_bbox(bbox, frame.shape[1], frame.shape[0])
                return True, self.bbox
        return self._recover_from_templates(frame)


def make_cv_tracker(name: str):
    candidates = []
    upper = name.upper()
    if hasattr(cv2, "legacy"):
        candidates.append((cv2.legacy, f"Tracker{upper}_create"))
    candidates.append((cv2, f"Tracker{upper}_create"))
    for module, factory in candidates:
        if hasattr(module, factory):
            return getattr(module, factory)()
    return None


def make_tracker(name: str):
    normalized = name.lower()
    if normalized == "template":
        return TemplateTracker()
    if normalized == "multi_template":
        return MultiTemplateTracker()
    tracker = make_cv_tracker(name)
    if tracker is None:
        print(f"tracker={name} unavailable; falling back to template", flush=True)
        return TemplateTracker()
    return tracker


def init_tracker(tracker, frame, bbox: BBox) -> bool:
    result = tracker.init(frame, tuple(bbox))
    return True if result is None else bool(result)


def update_tracker(tracker, frame):
    ok, bbox = tracker.update(frame)
    return bool(ok), bbox


def add_tracker_sample(tracker, frame, bbox: BBox) -> bool:
    add_sample = getattr(tracker, "add_sample", None)
    return bool(add_sample(frame, bbox)) if callable(add_sample) else False


def sanitize_bbox(bbox, width: int, height: int) -> BBox:
    x, y, w, h = [int(round(v)) for v in bbox]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def parse_labels(raw: str) -> list[str]:
    labels = [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]
    return labels or ["object"]


def resize_for_display(frame, scale: float):
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def scale_bbox(bbox: BBox, scale: float) -> BBox:
    x, y, w, h = bbox
    return int(x * scale), int(y * scale), int(w * scale), int(h * scale)


def color_for_index(index: int, ok: bool = True):
    palette = [
        (80, 220, 120),
        (80, 180, 255),
        (230, 120, 255),
        (255, 190, 70),
        (120, 220, 255),
        (180, 160, 255),
        (120, 255, 210),
        (255, 140, 140),
    ]
    color = palette[index % len(palette)]
    return color if ok else (0, 80, 255)


def draw_text(view, text: str, pos, scale: float = 0.58, color=(235, 245, 255)) -> None:
    cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def select_bbox(window: str, frame, scale: float) -> Optional[BBox]:
    shown = resize_for_display(frame, scale)
    roi = cv2.selectROI(window, shown, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window)
    if roi[2] <= 0 or roi[3] <= 0:
        return None
    inv = 1.0 / scale
    return sanitize_bbox((roi[0] * inv, roi[1] * inv, roi[2] * inv, roi[3] * inv), frame.shape[1], frame.shape[0])


def choose_initial_frame(cap, frame, args):
    window = "Choose initial frame"
    start_limit = max(0, args.start_frame)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    while True:
        view = frame.copy()
        total_text = f"/{total - 1}" if total > 0 else ""
        lines = [
            f"Choose clear initial frame: {frame_idx}{total_text}",
            f"a: back {args.seek_step} frames | d: forward {args.seek_step} frames",
            "Enter/Space: start box labeling | q/Esc: quit",
        ]
        y = 24
        for text in lines:
            draw_text(view, text, (12, y))
            y += 24
        cv2.imshow(window, resize_for_display(view, args.display_scale))
        key = cv2.waitKey(0) & 0xFF

        if key in (13, 10, ord(" ")):
            cv2.destroyWindow(window)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
            return frame, frame_idx
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise SystemExit("initial frame selection cancelled")
        if key in (ord("d"), 83):
            target = frame_idx + max(1, args.seek_step)
            if total > 0:
                target = min(total - 1, target)
        elif key in (ord("a"), 81):
            target = max(start_limit, frame_idx - max(1, args.seek_step))
        else:
            continue

        old_pos = frame_idx + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, next_frame = cap.read()
        if ok:
            frame = next_frame
            frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, old_pos)


def choose_label(labels: Sequence[str], frame, bbox: BBox, scale: float) -> Optional[str]:
    if len(labels) == 1:
        return labels[0]
    selected = 0
    window = "Choose label"
    while True:
        view = frame.copy()
        x, y, w, h = bbox
        cv2.rectangle(view, (x, y), (x + w, y + h), (80, 220, 120), 2)
        lines = ["Choose label for this box:"]
        for idx, label in enumerate(labels):
            prefix = "> " if idx == selected else "  "
            hotkey = str(idx + 1) if idx < 9 else " "
            lines.append(f"{prefix}{hotkey}. {label}")
        lines += ["Enter: confirm | n/p or +/-: switch | 1-9: quick choose | Esc: cancel"]
        y0 = 26
        for line in lines:
            draw_text(view, line, (12, y0), 0.56)
            y0 += 24
        cv2.imshow(window, resize_for_display(view, scale))
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.destroyWindow(window)
            return labels[selected]
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(labels):
                cv2.destroyWindow(window)
                return labels[idx]
        elif key in (ord("n"), ord("+"), ord("="), 83):
            selected = (selected + 1) % len(labels)
        elif key in (ord("p"), ord("-"), ord("_"), 81):
            selected = (selected - 1) % len(labels)


def create_track_object(frame, bbox: BBox, labels: Sequence[str], args, obj_id: int) -> Optional[TrackObject]:
    label = choose_label(labels, frame, bbox, args.display_scale)
    if label is None:
        return None
    tracker = make_tracker(args.tracker)
    if not init_tracker(tracker, frame, bbox):
        return None
    return TrackObject(obj_id=obj_id, label=label, bbox=bbox, tracker=tracker, ok=True)


def add_object_interactive(frame, labels: Sequence[str], args, obj_id: int, window_title: str = "Select object box, Enter confirms, Esc cancels") -> Optional[TrackObject]:
    bbox = select_bbox(window_title, frame, args.display_scale)
    if bbox is None:
        return None
    return create_track_object(frame, bbox, labels, args, obj_id)


def initial_objects(frame, labels: Sequence[str], args) -> list[TrackObject]:
    objects: list[TrackObject] = []
    next_id = 1
    while True:
        title = "Select object box, Enter confirms, Esc finishes"
        if objects:
            title = "Select next object, Enter confirms, Esc finishes"
        obj = add_object_interactive(frame, labels, args, next_id, title)
        if obj is None:
            break
        objects.append(obj)
        next_id += 1
    return objects


def draw_overlay(frame, objects: Sequence[TrackObject], active_idx: int, frame_idx: int, saved: int, interval: int, paused: bool):
    view = frame.copy()
    for idx, obj in enumerate(objects):
        x, y, w, h = obj.bbox
        color = color_for_index(idx, obj.ok)
        thickness = 3 if idx == active_idx else 2
        cv2.rectangle(view, (x, y), (x + w, y + h), color, thickness)
        label = f"#{obj.obj_id} {obj.label}"
        if obj.sample_count > 1:
            label += f" views:{obj.sample_count}"
        if not obj.ok:
            label += " LOST"
        draw_text(view, label, (x, max(18, y - 6)), 0.58, (245, 255, 245))
    ok_count = sum(1 for obj in objects if obj.ok)
    lines = [
        f"frame: {frame_idx}  saved: {saved}  interval: {interval}  objects: {ok_count}/{len(objects)}",
        "space: pause/resume | a: add box | r: reset active | v: add active view | tab/n: next | d: delete active",
        "s: save current | +/-: change interval | q/esc: quit",
    ]
    if objects:
        active = objects[active_idx]
        lines.append(f"active: #{active.obj_id} {active.label}  reference views: {active.sample_count}")
    if paused:
        lines.append("paused")
    if any(not obj.ok for obj in objects):
        lines.append("some tracking lost, press tab/n then r to fix or d to delete")
    y = 24
    for text in lines:
        draw_text(view, text, (12, y))
        y += 24
    return view


def write_voc_xml(xml_path: Path, image_name: str, width: int, height: int, depth: int, boxes: Sequence[tuple[str, BBox]]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"
    for label, bbox in boxes:
        x, y, w, h = bbox
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(x)
        ET.SubElement(box, "ymin").text = str(y)
        ET.SubElement(box, "xmax").text = str(x + w)
        ET.SubElement(box, "ymax").text = str(y + h)
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)


def save_sample(
    frame,
    objects: Sequence[TrackObject],
    args,
    frame_idx: int,
    saved: int,
    source_image_path: Optional[Path] = None,
) -> int:
    boxes = [(obj.label, obj.bbox) for obj in objects if obj.ok]
    if not boxes:
        return saved
    h, w = frame.shape[:2]
    depth = frame.shape[2] if len(frame.shape) == 3 else 1
    if source_image_path is not None:
        image_name = source_image_path.name
        xml_path = args.annotations_dir / f"{source_image_path.stem}.xml"
    else:
        stem = f"{args.prefix}_{frame_idx:06d}"
        image_name = stem + ".jpg"
        image_path = args.images_dir / image_name
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        xml_path = args.annotations_dir / (stem + ".xml")
    write_voc_xml(xml_path, image_name, w, h, depth, boxes)
    return saved + 1


def image_files_in(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted((path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions), key=lambda path: path.name.lower())


def choose_initial_image(image_files: Sequence[Path], start_idx: int, args):
    index = max(0, min(start_idx, len(image_files) - 1))
    window = "Choose initial image"
    while True:
        frame = cv2.imread(str(image_files[index]))
        if frame is None:
            raise SystemExit(f"failed to read image: {image_files[index]}")
        view = frame.copy()
        lines = [
            f"Choose clear initial image: {index}/{len(image_files) - 1}  {image_files[index].name}",
            f"a: back {args.seek_step} images | d: forward {args.seek_step} images",
            "Enter/Space: start box labeling | q/Esc: quit",
        ]
        y = 24
        for text in lines:
            draw_text(view, text, (12, y))
            y += 24
        cv2.imshow(window, resize_for_display(view, args.display_scale))
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, ord(" ")):
            cv2.destroyWindow(window)
            return frame, index
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise SystemExit("initial image selection cancelled")
        if key in (ord("d"), 83):
            index = min(len(image_files) - 1, index + max(1, args.seek_step))
        elif key in (ord("a"), 81):
            index = max(0, index - max(1, args.seek_step))



def parse_args():
    parser = argparse.ArgumentParser(description="Track one or more objects in a video or ordered image set and export VOC annotations.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="video path or camera index")
    source.add_argument("--images-input-dir", type=Path, help="ordered source image directory; XML files are written without copying images")
    parser.add_argument("--label", default="w", help="legacy single object class name; comma-separated values are also accepted")
    parser.add_argument("--labels", default="", help="comma/semicolon separated object class names, e.g. w,person,car")
    parser.add_argument("--interval", type=int, default=5, help="save one annotated frame every N frames/images")
    parser.add_argument("--images-dir", default="images", type=Path, help="video mode output image directory; ignored for image-set input")
    parser.add_argument("--annotations-dir", default="annotations", type=Path)
    parser.add_argument("--prefix", default="track", help="output filename prefix in video mode")
    parser.add_argument(
        "--tracker",
        default="csrt",
        choices=["csrt", "kcf", "mosse", "mil", "template", "multi_template"],
        help="multi_template supports multiple manually added reference views with the v key",
    )

    parser.add_argument("--start-frame", type=int, default=0, help="first video frame or image-set index")
    parser.add_argument("--seek-step", type=int, default=5, help="frames/images to move when choosing the initial item with a/d")
    parser.add_argument("--max-frames", type=int, default=0, help="maximum frames/images to process")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()




def open_capture(video: str):
    if not video.isdigit():
        return cv2.VideoCapture(video), None, "video"

    camera_index = int(video)
    backends = [(cv2.CAP_ANY, "default")]
    for backend, backend_name in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame, backend_name
            cv2.waitKey(30)
        cap.release()
    return None, None, ""


def choose_initial_camera_frame(cap, frame, args):
    window = "Confirm camera frame"
    while True:
        view = frame.copy()
        draw_text(view, "Confirm live camera view", (12, 24))
        draw_text(view, "Enter/Space: start box labeling | q/Esc: quit", (12, 48))
        cv2.imshow(window, resize_for_display(view, args.display_scale))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, ord(" ")):
            cv2.destroyWindow(window)
            return frame
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise SystemExit("camera frame confirmation cancelled")
        ok, next_frame = cap.read()
        if ok and next_frame is not None:
            frame = next_frame


def main():
    args = parse_args()
    args.annotations_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.labels or args.label)
    image_files: Optional[list[Path]] = None

    if args.images_input_dir is not None:
        if not args.images_input_dir.is_dir():
            raise SystemExit(f"image directory not found: {args.images_input_dir}")
        image_files = image_files_in(args.images_input_dir)
        if not image_files:
            raise SystemExit(f"no supported images found in: {args.images_input_dir}")
        frame, frame_idx = choose_initial_image(image_files, args.start_frame, args)
        source_image_path = image_files[frame_idx]
        source_mode = "image set"
    else:
        args.images_dir.mkdir(parents=True, exist_ok=True)
        is_camera = args.video.isdigit()
        cap, first_frame, capture_backend = open_capture(args.video)
        if cap is None or not cap.isOpened():
            if is_camera:
                raise SystemExit(
                    f"camera not available or cannot grab frames: {args.video}. "
                    "Close other camera apps, check Linux camera permissions, or try index 1/2."
                )
            raise SystemExit(f"video not readable: {args.video}")
        if is_camera:
            print(f"camera_index={args.video} backend={capture_backend}", flush=True)
            frame = choose_initial_camera_frame(cap, first_frame, args)
            frame_idx = 0
            source_mode = "camera"
        else:
            if args.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("failed to read first frame")
            frame, frame_idx = choose_initial_frame(cap, frame, args)
            source_mode = "video"
        source_image_path = None

    print("labels=" + ",".join(labels), flush=True)
    print(f"source={source_mode}", flush=True)
    print("Initial selection: press a/d to step backward/forward, Enter or Space starts labeling.", flush=True)
    print("Initial labeling: draw one box, choose its label, then repeat. Press Esc in ROI window when all initial objects are selected.", flush=True)
    if args.tracker == "multi_template":
        print("Multi-view tracking: pause on a new angle, select the active object, press v, and redraw that same object to add a reference view.", flush=True)
    objects = initial_objects(frame, labels, args)
    if not objects:
        raise SystemExit("no box selected")

    saved = save_sample(frame, objects, args, frame_idx, 0, source_image_path)
    processed = 0
    paused = False
    active_idx = 0
    next_obj_id = max(obj.obj_id for obj in objects) + 1
    if image_files is not None:
        window = "Image Set Track Label"
    elif source_mode == "camera":
        window = "Camera Track Label"
    else:
        window = "Video Track Label"

    while True:
        if not paused:
            if image_files is not None:
                next_idx = frame_idx + 1
                if next_idx >= len(image_files):
                    break
                frame_idx = next_idx
                source_image_path = image_files[frame_idx]
                frame = cv2.imread(str(source_image_path))
                if frame is None:
                    print(f"skipping unreadable image: {source_image_path}", flush=True)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                if source_mode == "camera":
                    frame_idx += 1
                else:
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                source_image_path = None
            processed += 1
            for obj in objects:
                if not obj.ok:
                    continue
                tracking_ok, next_bbox = update_tracker(obj.tracker, frame)
                obj.bbox = sanitize_bbox(next_bbox, frame.shape[1], frame.shape[0])
                obj.ok = tracking_ok

            if objects and args.interval > 0 and frame_idx % args.interval == 0:
                saved = save_sample(frame, objects, args, frame_idx, saved, source_image_path)
            if any(not obj.ok for obj in objects):
                paused = True

        if active_idx >= len(objects):
            active_idx = max(0, len(objects) - 1)
        view = draw_overlay(frame, objects, active_idx, frame_idx, saved, args.interval, paused)
        if image_files is not None:
            draw_text(view, f"image: {source_image_path.name}", (12, view.shape[0] - 14), 0.52)
        cv2.imshow(window, resize_for_display(view, args.display_scale))
        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (9, ord("n")) and objects:
            active_idx = (active_idx + 1) % len(objects)
        elif key == ord("p") and objects:
            active_idx = (active_idx - 1) % len(objects)
        elif key == ord("a"):
            obj = add_object_interactive(frame, labels, args, next_obj_id)
            if obj is not None:
                objects.append(obj)
                active_idx = len(objects) - 1
                next_obj_id += 1
                paused = False
        elif key == ord("r") and objects:
            new_bbox = select_bbox("Adjust active object box, Enter confirms, Esc cancels", frame, args.display_scale)
            if new_bbox is not None:
                obj = objects[active_idx]
                obj.bbox = new_bbox
                obj.tracker = make_tracker(args.tracker)
                obj.ok = init_tracker(obj.tracker, frame, obj.bbox)
                obj.sample_count = 1
                paused = False
        elif key == ord("v") and objects:
            new_bbox = select_bbox("Add active object view, Enter confirms, Esc cancels", frame, args.display_scale)
            if new_bbox is not None:
                obj = objects[active_idx]
                if add_tracker_sample(obj.tracker, frame, new_bbox):
                    obj.bbox = new_bbox
                    obj.ok = True
                    obj.sample_count += 1
                    paused = False
                else:
                    print("Adding reference views requires --tracker multi_template; use r to reset other trackers.", flush=True)
        elif key == ord("d") and objects:
            del objects[active_idx]
            active_idx = max(0, min(active_idx, len(objects) - 1))
            if not objects:
                paused = True
        elif key == ord("s"):
            saved = save_sample(frame, objects, args, frame_idx, saved, source_image_path)
        elif key in (ord("+"), ord("=")):
            args.interval = max(1, args.interval + 1)
        elif key in (ord("-"), ord("_")):
            args.interval = max(1, args.interval - 1)

        if args.max_frames > 0 and processed >= args.max_frames:
            break

    if image_files is None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"saved={saved}")
    if image_files is not None:
        print(f"images_dir={args.images_input_dir.resolve()}")
    else:
        print(f"images_dir={args.images_dir.resolve()}")
    print(f"annotations_dir={args.annotations_dir.resolve()}")



if __name__ == "__main__":
    main()
