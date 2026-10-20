"""实时检测 + 自动收集误报帧（难负样本）。

画面里只有蓝色物块时运行：一旦检出 orange/black/green（即误报），保存原始帧，
标注只保留高置信度的 blue 框，误报区域作为背景，用于下一轮训练抑制误报。
"""
import argparse
import collections
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from ultralytics import YOLO

from blue_guard import blue_mask, is_blue_target
from model_test import open_camera


def write_voc(path: Path, filename: str, width: int, height: int, boxes) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    for key, value in (("width", width), ("height", height), ("depth", 3)):
        ET.SubElement(size, key).text = str(value)
    ET.SubElement(root, "segmented").text = "0"
    for name, (x1, y1, x2, y2) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        for key in ("truncated", "difficult", "occluded"):
            ET.SubElement(obj, key).text = "0"
        box = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
            ET.SubElement(box, key).text = str(int(round(value)))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/liu/datasets/blocks_v1/outputs_20261001_141203/blocks_v3.pt")
    ap.add_argument("--camera-index", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.5, help="显示阈值")
    ap.add_argument("--save-conf", type=float, default=0.25, help="误报达到该置信度就保存帧")
    ap.add_argument("--keep-class", default="blue", help="画面里真实存在的类别，其框会保留到标注")
    ap.add_argument("--keep-conf", type=float, default=0.6, help="保留框的最低置信度；低于此的同类框视为不确定，整帧跳过")
    ap.add_argument("--min-interval", type=float, default=0.5, help="两次保存的最小间隔（秒），避免近似重复帧")
    ap.add_argument("--out", default="/home/liu/datasets/blocks_v1_hardneg")
    args = ap.parse_args()

    session = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / session
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    prefix = f"hardneg_{session}"  # 同一次运行算一个 video_group，划分时不会拆到 train/val 两边

    model = YOLO(args.model)
    cap, _ = open_camera(args.camera_index)
    if cap is None:
        raise SystemExit(f"camera {args.camera_index} not available")
    print(f"model={args.model} camera={args.camera_index} conf={args.conf} save_conf={args.save_conf} out={out}", flush=True)
    print("按 q / ESC 退出", flush=True)
    cv2.namedWindow("Live Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detect", 1280, 720)

    frame_no = saved = skipped = 0
    last_save = 0.0
    frames, hits, best = 0, collections.Counter(), collections.defaultdict(float)
    reasons = collections.Counter()
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed", flush=True)
            break
        frame_no += 1
        raw = frame.copy()
        r = model.predict(frame, conf=args.save_conf, verbose=False)[0]
        dets = [(r.names[int(c)], float(p), b) for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        shown = [d for d in dets if d[1] >= args.conf]

        false_pos = [d for d in dets if d[0] != args.keep_class]
        keep = [d for d in dets if d[0] == args.keep_class]
        now = time.time()
        if false_pos and now - last_save >= args.min_interval:
            if any(p < args.keep_conf for _, p, _ in keep):
                skipped += 1  # 蓝框置信度不够高，标注可能不准，不保存
            else:
                stem = f"{prefix}_{frame_no:06d}"
                h, w = raw.shape[:2]
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_voc(out / "annotations" / f"{stem}.xml", f"{stem}.jpg", w, h, [(n, b) for n, _, b in keep])
                saved += 1
                last_save = now
                reasons.update(n for n, _, _ in false_pos)

        frames += 1
        seen = set()
        for n, p, _ in shown:
            seen.add(n)
            best[n] = max(best[n], p)
        hits.update(seen)

        # 推理用 save_conf 跑，显示只画 >= conf 的框
        view = raw.copy()
        mask = blue_mask(raw)
        for n, p, (x1, y1, x2, y2) in shown:
            color = {"blue": (255, 120, 0), "orange": (0, 140, 255), "green": (0, 200, 0), "black": (60, 60, 60)}.get(n, (255, 255, 255))
            label = f"{n} {p:.2f}"
            if n != args.keep_class and is_blue_target(mask, (x1, y1, x2, y2)):
                color, label = (0, 0, 255), f"{label} VETO:blue"  # HSV 兜底：这个非蓝框其实是蓝块，不能夹
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(view, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(view, f"saved {saved}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Live Detect", view)

        if now - t0 >= 1.0:
            parts = [f"{n}={hits[n]}/{frames}({best[n]:.2f})" for n in ("blue", "orange", "green", "black") if hits[n]]
            print(time.strftime("%H:%M:%S"), f"fps={frames / (now - t0):.0f}", " ".join(parts) or "无检出", f"| saved={saved} skipped={skipped}", flush=True)
            frames, hits, best, t0 = 0, collections.Counter(), collections.defaultdict(float), now
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"done saved={saved} skipped={skipped} false_pos={dict(reasons)} out={out}", flush=True)


if __name__ == "__main__":
    main()
