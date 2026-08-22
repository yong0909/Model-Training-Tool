import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_results(results):
    return results[0].plot()


def prediction_summary(result) -> tuple[str, float]:
    probs = getattr(result, "probs", None)
    if probs is None:
        return "", 0.0
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    names = result.names
    label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])
    return label, confidence


def predict_image(model: YOLO, image_path: Path, conf: float):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"图片无法读取：{image_path}")
    results = model.predict(frame, conf=conf, verbose=False)
    return draw_results(results), prediction_summary(results[0])


def run_image(model: YOLO, image_path: Path, conf: float):
    annotated, (label, confidence) = predict_image(model, image_path, conf)
    out_path = image_path.with_name(image_path.stem + "_predict" + image_path.suffix)
    cv2.imwrite(str(out_path), annotated)
    if label:
        print(f"TEST_CLASSIFICATION={label} confidence={confidence:.4f}", flush=True)
    print(f"TEST_OUTPUT_IMAGE={out_path}", flush=True)
    cv2.imshow("YOLO Test", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder(model: YOLO, folder_path: Path, conf: float, output_dir: Optional[Path]):
    image_paths = sorted(path for path in folder_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"图片文件夹中未找到支持的图片：{folder_path}")
    output_dir = output_dir or folder_path / "predict_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, image_path in enumerate(image_paths, start=1):
        try:
            annotated, (label, confidence) = predict_image(model, image_path, conf)
            relative_path = image_path.relative_to(folder_path)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                raise ValueError("预测图片写入失败")
            rows.append((str(relative_path), label, f"{confidence:.6f}", str(output_path)))
            print(f"TEST_FOLDER_PROGRESS={index}/{len(image_paths)} image={image_path} label={label or '-'}", flush=True)
        except Exception as exc:
            failures.append((str(image_path), str(exc)))
            print(f"TEST_FOLDER_ERROR={image_path} error={exc}", flush=True)

    summary_path = output_dir / "predictions.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "classification", "confidence", "output"))
        writer.writerows(rows)
    print(f"TEST_OUTPUT_DIR={output_dir}", flush=True)
    print(f"TEST_OUTPUT_SUMMARY={summary_path}", flush=True)
    print(f"TEST_FOLDER_COMPLETE=success:{len(rows)} failed:{len(failures)}", flush=True)
    if not rows:
        raise SystemExit("图片文件夹测试失败，未生成任何预测结果。")


def open_camera(camera_index: int):
    backends = [(cv2.CAP_ANY, "default")]

    for backend, name in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name
            cv2.waitKey(30)
        cap.release()
    return None, ""


def run_camera(model: YOLO, camera_index: int, conf: float):
    cap, backend_name = open_camera(camera_index)
    if cap is None:
        raise SystemExit(
            f"camera not available or cannot grab frames: {camera_index}. "
            "Please close other camera apps, check Linux camera permissions, "
            "or try --camera-index 1/2."
        )

    print(f"Camera opened with {backend_name}. Press q or ESC to quit.", flush=True)
    cv2.namedWindow("YOLO Camera Test", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera frame grab failed, exiting.", flush=True)
            break
        results = model.predict(frame, conf=conf, verbose=False)
        annotated = draw_results(results)
        cv2.imshow("YOLO Camera Test", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="测试 Ultralytics 检测或分类 .pt 模型")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", choices=["camera", "image", "folder"], required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--folder", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"模型不存在：{model_path}")
    model = YOLO(str(model_path))

    if args.source == "camera":
        run_camera(model, args.camera_index, args.conf)
    elif args.source == "image":
        if not args.image:
            raise SystemExit("--image is required when --source image")
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise SystemExit(f"图片不存在：{image_path}")
        run_image(model, image_path, args.conf)
    else:
        if not args.folder:
            raise SystemExit("--folder is required when --source folder")
        folder_path = Path(args.folder).resolve()
        if not folder_path.is_dir():
            raise SystemExit(f"图片文件夹不存在：{folder_path}")
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        run_folder(model, folder_path, args.conf, output_dir)


if __name__ == "__main__":
    main()
