# -*- coding: utf-8 -*-
"""Convert a static Ultralytics YOLO ONNX model to RKNN and build a board package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CALIBRATION_SEED = 3588


class DeployError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def emit(key: str, value: Any = "") -> None:
    print(f"{key}={value}", flush=True)


def stage(name: str) -> None:
    emit("RKNN_STAGE", name)


def package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "未安装"


def environment_info(require_toolkit: bool = True) -> dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "rknn_toolkit2": package_version("rknn-toolkit2", "rknn_toolkit2"),
        "onnx": package_version("onnx"),
        "numpy": package_version("numpy"),
    }
    try:
        from rknn.api import RKNN  # noqa: F401

        info["toolkit_importable"] = True
    except Exception as exc:
        info["toolkit_importable"] = False
        info["toolkit_error"] = str(exc)
        if require_toolkit:
            raise DeployError(
                "environment",
                "无法导入 rknn.api.RKNN。请使用板卡厂商 SDK 配套的独立 Python 环境，"
                f"并安装对应 rknn_toolkit2 wheel。原始错误：{exc}",
            ) from exc
    return info


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise DeployError("classes", f"类别文件不存在：{path}")
    try:
        classes = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except OSError as exc:
        raise DeployError("classes", f"无法读取类别文件：{path}；{exc}") from exc
    if not classes:
        raise DeployError("classes", "classes.txt 不能为空，至少需要一个类别名称。")
    return classes


def onnx_dimension(dim: Any) -> Optional[int]:
    value = int(getattr(dim, "dim_value", 0) or 0)
    return value if value > 0 else None


def shape_text(shape: list[Optional[int]]) -> str:
    return ",".join(str(value) if value is not None else "?" for value in shape)


def inspect_onnx(onnx_path: Path, classes_path: Path) -> dict[str, Any]:
    stage("check_onnx")
    if not onnx_path.is_file():
        raise DeployError("onnx", f"ONNX 模型不存在：{onnx_path}")
    if onnx_path.suffix.lower() != ".onnx":
        raise DeployError("onnx", f"模型文件必须使用 .onnx 扩展名：{onnx_path}")
    try:
        import onnx
    except Exception as exc:
        raise DeployError("environment", f"当前转换环境无法导入 onnx：{exc}") from exc

    classes = load_classes(classes_path)
    try:
        model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise DeployError(
            "onnx",
            "ONNX 文件读取或结构检查失败。若 RKNN-Toolkit2 不支持当前模型，请从 .pt 重新导出静态 "
            f"opset=12 ONNX，无需重新训练。原始错误：{exc}",
        ) from exc

    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(inputs) != 1:
        raise DeployError("onnx", f"只支持一个图片输入，当前模型检测到 {len(inputs)} 个运行时输入。")

    input_value = inputs[0]
    input_shape = [onnx_dimension(dim) for dim in input_value.type.tensor_type.shape.dim]
    if len(input_shape) != 4 or any(value is None for value in input_shape):
        raise DeployError("onnx", f"输入必须是固定四维张量，当前形状为 [{shape_text(input_shape)}]。")
    fixed_shape = [int(value) for value in input_shape if value is not None]
    if fixed_shape[0] != 1:
        raise DeployError("onnx", f"仅支持 batch=1 的静态模型，当前 batch={fixed_shape[0]}。")
    if fixed_shape[1] in {1, 3, 4}:
        layout, channels, height, width = "NCHW", fixed_shape[1], fixed_shape[2], fixed_shape[3]
    elif fixed_shape[3] in {1, 3, 4}:
        layout, channels, height, width = "NHWC", fixed_shape[3], fixed_shape[1], fixed_shape[2]
    else:
        raise DeployError("onnx", f"无法识别图片通道维度，当前输入形状为 [{shape_text(input_shape)}]。")
    if channels != 3:
        raise DeployError("onnx", f"板端示例仅支持 RGB 三通道输入，当前通道数为 {channels}。")
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise DeployError("onnx", f"输入宽高必须不小于 32 且为 32 的倍数，当前为 {width}x{height}。")

    output_shapes: list[list[Optional[int]]] = []
    for output in model.graph.output:
        output_shapes.append([onnx_dimension(dim) for dim in output.type.tensor_type.shape.dim])
    if not output_shapes:
        raise DeployError("onnx", "ONNX 模型没有输出张量。")
    expected_channels = {len(classes) + 4, len(classes) + 5}
    fixed_output_dims = {value for shape in output_shapes for value in shape if value is not None}
    if not expected_channels.intersection(fixed_output_dims):
        expected = " 或 ".join(str(value) for value in sorted(expected_channels))
        raise DeployError(
            "onnx",
            f"输出形状与 {len(classes)} 个类别不匹配：未找到检测通道维度 {expected}。"
            "第一版仅支持 Ultralytics YOLO 目标检测的原始输出。",
        )

    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    opset = int(opsets.get("ai.onnx", 0))
    metadata = {
        "onnx_path": str(onnx_path.resolve()),
        "classes_path": str(classes_path.resolve()),
        "classes": classes,
        "class_count": len(classes),
        "opset": opset,
        "opsets": opsets,
        "input_name": input_value.name,
        "input_shape": fixed_shape,
        "input_layout": layout,
        "input_width": width,
        "input_height": height,
        "output_names": [item.name for item in model.graph.output],
        "output_shapes": output_shapes,
    }
    emit("RKNN_INPUT_NAME", input_value.name)
    emit("RKNN_INPUT_SIZE", f"{width}x{height}")
    for output_shape in output_shapes:
        emit("RKNN_OUTPUT_SHAPE", shape_text(output_shape))
    emit("RKNN_ONNX_OPSET", opset)
    emit("RKNN_CLASS_COUNT", len(classes))
    return metadata


def calibration_images(directory: Path, count: int) -> list[Path]:
    if not directory.is_dir():
        raise DeployError("calibration", f"INT8 校准图片目录不存在：{directory}")
    images = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stat().st_size > 0
    )
    if not images:
        raise DeployError("calibration", "校准目录中没有可用的 JPG、PNG、BMP 或 WEBP 图片。")
    rng = random.Random(CALIBRATION_SEED)
    selected = rng.sample(images, min(count, len(images)))
    if len(selected) < 20:
        raise DeployError("calibration", f"有效校准图片只有 {len(selected)} 张，至少需要 20 张。")
    if len(selected) < 100:
        print(f"警告：有效校准图片只有 {len(selected)} 张，建议至少使用 100 张以降低量化精度损失。", flush=True)
    return selected


def require_success(result: Any, operation: str) -> None:
    if result not in (None, 0):
        raise DeployError("conversion", f"RKNN {operation} 失败，返回码：{result}")


def build_rknn(
    onnx_path: Path,
    output_path: Path,
    quantized: bool,
    dataset_path: Optional[Path] = None,
) -> None:
    try:
        from rknn.api import RKNN
    except Exception as exc:
        raise DeployError("environment", f"无法导入 rknn.api.RKNN：{exc}") from exc

    rknn = RKNN(verbose=True)
    try:
        require_success(
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform="rk3588",
            ),
            "config",
        )
        require_success(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        build_args: dict[str, Any] = {"do_quantization": quantized}
        if quantized:
            if dataset_path is None:
                raise DeployError("calibration", "INT8 构建缺少校准列表。")
            build_args["dataset"] = str(dataset_path)
        require_success(rknn.build(**build_args), "build")
        require_success(rknn.export_rknn(str(output_path)), "export_rknn")
    except DeployError:
        raise
    except Exception as exc:
        hint = "若日志提示 opset 或算子不支持，请从 .pt 重新导出静态 opset=12 ONNX，无需重新训练。"
        raise DeployError("conversion", f"RKNN 转换失败：{exc}。{hint}") from exc
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def simulated_inference(rknn_path: Path, image_path: Path, width: int, height: int) -> list[list[int]]:
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise DeployError("test", f"测试图片无效：{image_path}")
    try:
        import cv2
        import numpy as np
        from rknn.api import RKNN
    except Exception as exc:
        raise DeployError("environment", f"模拟推理需要 RKNN、NumPy 和 OpenCV：{exc}") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise DeployError("test", f"OpenCV 无法读取测试图片：{image_path}")
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height))
    input_data = resized.astype(np.uint8)
    rknn = RKNN(verbose=False)
    try:
        require_success(rknn.load_rknn(str(rknn_path)), "load_rknn")
        require_success(rknn.init_runtime(), "init_runtime")
        outputs = rknn.inference(inputs=[input_data], data_format=["nhwc"])
        if not outputs:
            raise DeployError("test", "模拟推理没有返回输出张量。")
        shapes = [list(np.asarray(item).shape) for item in outputs]
        print(f"模拟推理通过，输出形状：{shapes}", flush=True)
        return shapes
    except DeployError:
        raise
    except Exception as exc:
        raise DeployError("test", f"转换后模拟推理失败：{exc}") from exc
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


BOARD_INFERENCE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def letterbox(image, size):
    target_w, target_h = size
    height, width = image.shape[:2]
    ratio = min(target_w / width, target_h / height)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = target_w - new_w, target_h - new_h
    left, top = pad_w // 2, pad_h // 2
    canvas = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, ratio, (left, top)


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def nms(boxes, scores, threshold):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[current], boxes[remaining]) <= threshold]
    return keep


def detection_rows(outputs, class_count):
    rows = []
    expected = {class_count + 4, class_count + 5}
    for output in outputs:
        array = np.asarray(output).squeeze()
        if array.ndim != 2:
            continue
        if array.shape[0] in expected:
            rows.append(array.T)
        elif array.shape[1] in expected:
            rows.append(array)
    if not rows:
        shapes = [list(np.asarray(item).shape) for item in outputs]
        raise RuntimeError(f"不支持的 YOLO 输出形状：{shapes}")
    return np.concatenate(rows, axis=0)


def decode(outputs, class_count, confidence, iou_threshold):
    predictions = detection_rows(outputs, class_count)
    boxes_xywh = predictions[:, :4].astype(np.float32)
    if predictions.shape[1] == class_count + 5:
        class_scores = predictions[:, 5:] * predictions[:, 4:5]
    else:
        class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    valid = scores >= confidence
    boxes_xywh, scores, class_ids = boxes_xywh[valid], scores[valid], class_ids[valid]
    if not len(scores):
        return np.empty((0, 4)), scores, class_ids
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    selected = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        selected.extend(indices[nms(boxes[indices], scores[indices], iou_threshold)])
    selected = np.asarray(selected, dtype=np.int64)
    return boxes[selected], scores[selected], class_ids[selected]


def main():
    parser = argparse.ArgumentParser(description="RK3588 RKNN 单图片目标检测")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="result.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manifest = __import__("json").loads((root / "deployment_manifest.json").read_text(encoding="utf-8"))
    classes = manifest["classes"]
    width, height = manifest["input"]["width"], manifest["input"]["height"]
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"无法读取图片：{args.image}")
    prepared, ratio, (pad_x, pad_y) = letterbox(image, (width, height))
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)

    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        raise SystemExit("加载 RKNN 模型失败，请核对板端 Runtime 与转换 Toolkit 版本。")
    if rknn.init_runtime() != 0:
        raise SystemExit("初始化 RKNN Runtime 失败，请核对 RKNPU 驱动和 Runtime。")
    try:
        outputs = rknn.inference(inputs=[rgb.astype(np.uint8)])
    finally:
        rknn.release()

    boxes, scores, class_ids = decode(outputs, len(classes), args.conf, args.iou)
    if len(boxes):
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.shape[1] - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.shape[0] - 1)
    for box, score, class_id in zip(boxes.astype(int), scores, class_ids):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (32, 220, 120), 2)
        cv2.putText(image, f"{classes[int(class_id)]} {score:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (32, 220, 120), 2, cv2.LINE_AA)
    if not cv2.imwrite(args.output, image):
        raise SystemExit(f"结果图片保存失败：{args.output}")
    print(f"detections={len(boxes)}")
    print(f"output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
'''


BOARD_README = '''# RK3588 / RK3588S 单图片目标检测包

本目录由 `model-training-tool` 生成，只负责转换与打包，不包含板卡驱动、
`librknnrt.so` 或 `rknn-toolkit-lite2` wheel。

## 准备板端环境

1. 从板卡厂商 SDK 或系统镜像获取与 RKNPU 驱动匹配的 RKNN Runtime 和 Lite2 wheel。
2. 安装厂商提供的 `rknn_toolkit_lite2-*.whl`，再安装 `requirements-board.txt` 中其余依赖。
3. Runtime/Lite2 版本必须与生成模型所用 RKNN-Toolkit2 兼容；不要盲目安装最新版。

## 运行

```bash
python3 infer_image.py --model models/model_fp.rknn --image test.jpg --output result.jpg
```

如已生成 INT8 模型，可把 `--model` 改为 `models/model_int8.rknn`。
板端示例执行 letterbox、BGR 转 RGB；归一化由模型中的 mean/std 配置完成，
请勿在输入侧再次除以 255。

第一版仅支持 Ultralytics YOLO 目标检测的单个原始预测输出。
'''


def create_package(
    output_dir: Path,
    metadata: dict[str, Any],
    environment: dict[str, Any],
    mode: str,
    fp_model: Optional[Path],
    int8_model: Optional[Path],
    classes_path: Path,
    calibration_path: Optional[Path],
    test_image: Optional[Path],
    conf: float,
    iou: float,
    calibration_count: int,
) -> tuple[Path, Path]:
    stage("package")
    manifest_path = output_dir / "deployment_manifest.json"
    model_files = [path.name for path in (fp_model, int8_model) if path is not None]
    manifest = {
        "format_version": 1,
        "target_platform": "rk3588",
        "compatible_boards": ["RK3588", "RK3588S"],
        "source_onnx": {
            "file": Path(metadata["onnx_path"]).name,
            "sha256": sha256_file(Path(metadata["onnx_path"])),
            "opset": metadata["opset"],
        },
        "input": {
            "name": metadata["input_name"],
            "shape": metadata["input_shape"],
            "source_layout": metadata["input_layout"],
            "board_layout": "NHWC",
            "dtype": "uint8",
            "color": "RGB",
            "width": metadata["input_width"],
            "height": metadata["input_height"],
            "mean": [0, 0, 0],
            "std": [255, 255, 255],
        },
        "outputs": [
            {"name": name, "shape": shape}
            for name, shape in zip(metadata["output_names"], metadata["output_shapes"])
        ],
        "classes": metadata["classes"],
        "class_count": metadata["class_count"],
        "rknn_toolkit2": environment["rknn_toolkit2"],
        "conversion_mode": mode,
        "calibration_image_count": calibration_count,
        "default_confidence": conf,
        "default_nms_iou": iou,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_files": model_files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    package_path = output_dir / "rk3588_deploy.zip"
    try:
        with tempfile.TemporaryDirectory(prefix="rk3588-package-") as temporary:
            root = Path(temporary) / "rk3588_yolo"
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            for model_path in (fp_model, int8_model):
                if model_path is not None:
                    shutil.copy2(model_path, models_dir / model_path.name)
            shutil.copy2(classes_path, root / "classes.txt")
            shutil.copy2(manifest_path, root / manifest_path.name)
            (root / "infer_image.py").write_text(BOARD_INFERENCE, encoding="utf-8")
            (root / "requirements-board.txt").write_text("numpy\nopencv-python\n", encoding="utf-8")
            (root / "README_RK3588.md").write_text(BOARD_README, encoding="utf-8")
            if calibration_path is not None:
                shutil.copy2(calibration_path, root / "calibration.txt")
            if test_image is not None:
                shutil.copy2(test_image, root / "test.jpg")
            with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(Path(temporary)))
    except Exception as exc:
        raise DeployError("package", f"生成部署包失败：{exc}") from exc
    return package_path, manifest_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Ultralytics YOLO ONNX 转换为 RK3588 RKNN 并生成部署包")
    parser.add_argument("--onnx", help="静态 Ultralytics YOLO 目标检测 ONNX")
    parser.add_argument("--classes", help="类别文件，每行一个类别")
    parser.add_argument("--output-dir", help="输出目录；留空时在 ONNX 同目录创建时间戳目录")
    parser.add_argument("--mode", choices=("fp", "int8", "both"), default="both")
    parser.add_argument("--calibration-dir", help="INT8 校准图片目录，递归读取")
    parser.add_argument("--calibration-count", type=int, default=200)
    parser.add_argument("--test-image", help="可选的转换后模拟推理图片")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--check-environment", action="store_true", help="只检查当前 RKNN Python 环境")
    parser.add_argument("--check-only", action="store_true", help="只检查 ONNX 和类别文件")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.check_environment:
        return
    if not args.onnx:
        raise DeployError("arguments", "缺少 --onnx 模型路径。")
    if not args.classes:
        raise DeployError("arguments", "缺少 --classes 类别文件路径。")
    if args.check_only:
        return
    if not 20 <= args.calibration_count <= 1000:
        raise DeployError("arguments", "--calibration-count 必须在 20 到 1000 之间。")
    if not 0 < args.conf <= 1:
        raise DeployError("arguments", "--conf 必须大于 0 且不超过 1。")
    if not 0 < args.iou <= 1:
        raise DeployError("arguments", "--iou 必须大于 0 且不超过 1。")
    if args.mode in {"int8", "both"} and not args.calibration_dir:
        raise DeployError("calibration", "INT8 模式必须提供 --calibration-dir。")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.check_environment:
        stage("check_environment")
        info = environment_info(require_toolkit=True)
        emit("RKNN_ENVIRONMENT_JSON", json.dumps(info, ensure_ascii=False, separators=(",", ":")))
        print("环境检查通过。请确认 Toolkit2 版本与开发板 RKNN Runtime/RKNPU 驱动匹配。", flush=True)
        return

    onnx_path = Path(args.onnx).expanduser().resolve()
    classes_path = Path(args.classes).expanduser().resolve()
    if args.check_only:
        metadata = inspect_onnx(onnx_path, classes_path)
        emit("RKNN_MODEL_INFO_JSON", json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
        print("ONNX 模型检查通过。", flush=True)
        return

    stage("check_environment")
    environment = environment_info(require_toolkit=True)
    emit("RKNN_ENVIRONMENT_JSON", json.dumps(environment, ensure_ascii=False, separators=(",", ":")))
    metadata = inspect_onnx(onnx_path, classes_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else onnx_path.parent / f"rk3588_deploy_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_images: list[Path] = []
    calibration_path: Optional[Path] = None
    if args.mode in {"int8", "both"}:
        selected_images = calibration_images(Path(args.calibration_dir).expanduser().resolve(), args.calibration_count)
        calibration_path = output_dir / "calibration.txt"
        calibration_path.write_text("\n".join(str(path) for path in selected_images) + "\n", encoding="utf-8")
        emit("RKNN_CALIBRATION_COUNT", len(selected_images))

    fp_model: Optional[Path] = None
    int8_model: Optional[Path] = None
    if args.mode in {"fp", "both"}:
        stage("build_fp")
        fp_model = output_dir / "model_fp.rknn"
        build_rknn(onnx_path, fp_model, quantized=False)
        emit("RKNN_MODEL_FP", fp_model)
        if args.test_image:
            simulated_inference(fp_model, Path(args.test_image).expanduser().resolve(), metadata["input_width"], metadata["input_height"])

    if args.mode in {"int8", "both"}:
        stage("build_int8")
        int8_model = output_dir / "model_int8.rknn"
        try:
            build_rknn(onnx_path, int8_model, quantized=True, dataset_path=calibration_path)
            emit("RKNN_MODEL_INT8", int8_model)
            if args.test_image:
                simulated_inference(int8_model, Path(args.test_image).expanduser().resolve(), metadata["input_width"], metadata["input_height"])
        except DeployError as exc:
            if fp_model is not None and fp_model.is_file():
                emit("RKNN_PARTIAL_SUCCESS", "fp_available_int8_failed")
                print(f"非量化模型可用：{fp_model}；INT8 转换失败。", flush=True)
            raise exc

    test_image = Path(args.test_image).expanduser().resolve() if args.test_image else None
    package_path, manifest_path = create_package(
        output_dir=output_dir,
        metadata=metadata,
        environment=environment,
        mode=args.mode,
        fp_model=fp_model,
        int8_model=int8_model,
        classes_path=classes_path,
        calibration_path=calibration_path,
        test_image=test_image,
        conf=args.conf,
        iou=args.iou,
        calibration_count=len(selected_images),
    )
    emit("RKNN_PACKAGE", package_path)
    emit("RKNN_MANIFEST", manifest_path)
    stage("complete")
    print("RK3588 转换与部署包生成完成。", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except DeployError as exc:
        emit("RKNN_ERROR_TYPE", exc.category)
        emit("RKNN_ERROR", str(exc))
        print(f"错误：{exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        emit("RKNN_ERROR_TYPE", "stopped")
        emit("RKNN_ERROR", "用户已停止转换。")
        return 130
    except Exception as exc:
        emit("RKNN_ERROR_TYPE", "unexpected")
        emit("RKNN_ERROR", f"未预期错误：{exc}")
        print(f"未预期错误：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
