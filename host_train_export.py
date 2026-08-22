from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import zipfile
from pathlib import Path

from PIL import Image



def configure_stdio():
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





def subprocess_env():
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run(cmd, cwd=None, check=True, env=None, timeout=None):
    argv = [str(x) for x in cmd]
    print("\n$ " + " ".join(argv), flush=True)
    started_at = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            env=env or subprocess_env(),
            stdin=sys.stdin,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started_at
        print(f"COMMAND_TIMEOUT elapsed_seconds={elapsed:.1f} timeout_seconds={timeout}", flush=True)
        raise SystemExit(f"Command timed out after {timeout} seconds: {argv[0]}") from exc
    elapsed = time.monotonic() - started_at
    print(f"COMMAND_EXIT_CODE={result.returncode} elapsed_seconds={elapsed:.1f}", flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, argv)
    return result


def interrupt_process_tree(proc: subprocess.Popen, timeout: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def run_train_command(cmd, stop_signal: Path | None = None, cwd=None):
    argv = [str(x) for x in cmd]
    print("\n$ " + " ".join(argv), flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=subprocess_env(),
        stdin=sys.stdin,
        start_new_session=True,
    )
    while True:
        code = proc.poll()
        if code is not None:
            if code != 0:
                print(f"TRAIN_PROCESS_EXIT_CODE={code}", flush=True)
            return code
        if stop_signal and stop_signal.exists():
            print("TRAIN_STOP_EXPORT_REQUESTED=1", flush=True)
            interrupt_process_tree(proc)
            return proc.poll() if proc.poll() is not None else 130
        time.sleep(1.0)


def torch_index_url(torch_cuda):

    torch_cuda = (torch_cuda or "").strip().lower()
    if torch_cuda in {"cu118", "cu121", "cu124", "cu128", "cpu"}:
        return f"https://download.pytorch.org/whl/{torch_cuda}"
    return ""


def torch_install_args(torch_cuda):
    index_url = torch_index_url(torch_cuda)
    if not index_url:
        return []
    return ["torch", "torchvision", "torchaudio", "--index-url", index_url]


def build_env_cmd(conda_env, exe):

    conda_env = conda_env.strip()
    current_env = (os.environ.get("CONDA_DEFAULT_ENV") or "").strip()

    if conda_env and current_env.lower() == conda_env.lower():
        if exe == "python":
            return [sys.executable]
        found = shutil.which(exe)
        if found:
            return [found]
        scripts_exe = Path(sys.executable).parent / "Scripts" / (exe + ".exe")
        if scripts_exe.exists():
            return [str(scripts_exe)]
        return [exe]

    conda = shutil.which("conda")
    if conda_env and conda:
        return [conda, "run", "--no-capture-output", "-n", conda_env, exe]
    if exe == "python":
        return [sys.executable]
    found = shutil.which(exe)
    if found:
        return [found]
    scripts_exe = Path(sys.executable).parent / "Scripts" / (exe + ".exe")
    if scripts_exe.exists():
        return [str(scripts_exe)]
    return [exe]


def parse_train_ratio_percent(value):
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise SystemExit("--train-ratio-percent 必须是 1 到 100 的数字")
    if not 1 <= ratio <= 100:
        raise SystemExit("--train-ratio-percent 必须在 1 到 100 之间")
    return ratio


def validate_image_dimensions(args):
    legacy_size = args.img_size
    if args.img_width is None:
        args.img_width = legacy_size or 448
    if args.img_height is None:
        args.img_height = legacy_size or 448
    for option, value in (("--img-width", args.img_width), ("--img-height", args.img_height)):
        if value < 32 or value % 32:
            raise SystemExit(f"{option} 必须是大于等于 32 的 32 倍数")


def imgsz_arg(args):
    return f"{args.img_height},{args.img_width}"


def clip_box(box, width, height):
    name, xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(float(width), xmin))
    ymin = max(0.0, min(float(height), ymin))
    xmax = max(0.0, min(float(width), xmax))
    ymax = max(0.0, min(float(height), ymax))
    return (name, xmin, ymin, xmax, ymax) if xmax > xmin and ymax > ymin else None


def transform_image_and_boxes(image_path: Path, boxes, source_size, target_size, resize_mode):
    source_width, source_height = source_size
    target_width, target_height = target_size
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    if image.size != (source_width, source_height):
        raise SystemExit(f"图片尺寸与 XML 标注不一致: {image_path}，图片={image.size}，XML={source_size}")

    if resize_mode == "crop":
        target_ratio = target_width / target_height
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = round(source_height * target_ratio)
            left, top = (source_width - crop_width) // 2, 0
            right, bottom = left + crop_width, source_height
        else:
            crop_height = round(source_width / target_ratio)
            left, top = 0, (source_height - crop_height) // 2
            right, bottom = source_width, top + crop_height
        image = image.crop((left, top, right, bottom)).resize((target_width, target_height), Image.Resampling.LANCZOS)
        scale_x = target_width / (right - left)
        scale_y = target_height / (bottom - top)
        transformed = []
        for name, xmin, ymin, xmax, ymax in boxes:
            clipped = clip_box((name, xmin - left, ymin - top, xmax - left, ymax - top), right - left, bottom - top)
            if clipped:
                name, xmin, ymin, xmax, ymax = clipped
                transformed.append((name, xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y))
    elif resize_mode == "stretch":
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        transformed = [(name, xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y) for name, xmin, ymin, xmax, ymax in boxes]
    else:
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = round(source_width * scale)
        resized_height = round(source_height * scale)
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        image = Image.new("RGB", (target_width, target_height), (114, 114, 114))
        image.paste(resized, (pad_left, pad_top))
        transformed = [(name, xmin * scale + pad_left, ymin * scale + pad_top, xmax * scale + pad_left, ymax * scale + pad_top) for name, xmin, ymin, xmax, ymax in boxes]

    return image, [box for box in (clip_box(box, target_width, target_height) for box in transformed) if box]


def prepare_voc_yolo(images_dir: Path, annotations_dir: Path, out: Path, train_ratio_percent=80, seed=42, img_width=448, img_height=448, resize_mode="letterbox"):
    train_ratio_percent = parse_train_ratio_percent(train_ratio_percent)
    val_ratio = max(0.0, min(0.99, (100.0 - train_ratio_percent) / 100.0))
    img_dir = images_dir.resolve()
    ann_dir = annotations_dir.resolve()
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit("Images dir and annotations dir must both exist")
    if resize_mode not in {"crop", "letterbox", "stretch"}:
        raise SystemExit("--image-resize-mode 必须是 crop、letterbox 或 stretch")

    xmls = sorted(ann_dir.glob("*.xml"))
    classes = []
    records = []
    for xml_path in xmls:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        size = root.find("size")
        if size is None:
            continue
        width = int(float(size.findtext("width")))
        height = int(float(size.findtext("height")))
        if width <= 0 or height <= 0:
            continue
        boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if not name or box is None:
                continue
            parsed = clip_box((name, float(box.findtext("xmin")), float(box.findtext("ymin")), float(box.findtext("xmax")), float(box.findtext("ymax"))), width, height)
            if parsed:
                if name not in classes:
                    classes.append(name)
                boxes.append(parsed)
        image_path = img_dir / filename
        if image_path.exists() and boxes:
            records.append((image_path, (width, height), boxes))

    if not records:
        raise SystemExit("No valid image/xml records found")
    if len(records) < 2:
        raise SystemExit("至少需要 2 张有效标注图片，才能划分训练集和验证集")

    random.seed(seed)
    random.shuffle(records)
    val_n = int(round(len(records) * val_ratio))
    val_n = max(1, min(val_n, len(records) - 1))
    splits = {"val": records[:val_n], "train": records[val_n:]}
    class_ids = {name: index for index, name in enumerate(classes)}

    output_counts = {}
    for split, items in splits.items():
        image_output_dir = out / "images" / split
        label_output_dir = out / "labels" / split
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for image_path, source_size, boxes in items:
            image, boxes = transform_image_and_boxes(image_path, boxes, source_size, (img_width, img_height), resize_mode)
            if not boxes:
                continue
            output_stem = image_path.stem
            image.save(image_output_dir / f"{output_stem}.jpg", quality=95)
            lines = []
            for name, xmin, ymin, xmax, ymax in boxes:
                xc = ((xmin + xmax) / 2) / img_width
                yc = ((ymin + ymax) / 2) / img_height
                bw = (xmax - xmin) / img_width
                bh = (ymax - ymin) / img_height
                lines.append(f"{class_ids[name]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
            (label_output_dir / f"{output_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
        output_counts[split] = written
    if not output_counts.get("train") or not output_counts.get("val"):
        raise SystemExit("当前裁剪方式移除了训练集或验证集的全部标注框；请改用等比缩放或拉伸。")

    dataset_yaml = out / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(classes)),
        encoding="utf-8",
    )
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"classes={classes}")
    print(f"image_resize_mode={resize_mode} target_size={img_width}x{img_height}")
    print(f"train={len(splits['train'])} val={len(splits['val'])}")
    return dataset_yaml, classes


def prepare_classification_yolo(images_dir: Path, out: Path, train_ratio_percent=80, seed=42):
    """按“图片根目录/类别名/图片”结构生成 Ultralytics 分类训练集。"""
    train_ratio_percent = parse_train_ratio_percent(train_ratio_percent)
    source_root = images_dir.resolve()
    if not source_root.is_dir():
        raise SystemExit("分类 Images Dir 必须是有效文件夹，且下级目录为类别名")

    classes = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise SystemExit("分类数据集至少需要 2 个类别子文件夹，例如 images/正常、images/异常")

    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    randomizer = random.Random(seed)
    counts = {}
    for class_name in classes:
        images = sorted(
            path for path in (source_root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in allowed_extensions
        )
        if len(images) < 2:
            raise SystemExit(f"分类 {class_name} 至少需要 2 张图片，才能划分训练集和验证集")
        randomizer.shuffle(images)
        val_count = max(1, min(int(round(len(images) * (100.0 - train_ratio_percent) / 100.0)), len(images) - 1))
        splits = {"val": images[:val_count], "train": images[val_count:]}
        for split, items in splits.items():
            target_dir = out / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(items):
                # 同类的重复文件名用序号区分，避免覆盖。
                target_name = f"{index:06d}_{image_path.name}"
                shutil.copy2(image_path, target_dir / target_name)
        counts[class_name] = {split: len(items) for split, items in splits.items()}

    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"classes={classes}")
    print(f"classification_split={counts}")
    return out, classes


def zip_dir_contents(src_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir).as_posix())


def ps_single_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def ssh_base_args(port):
    args = ["ssh"]
    if int(port) != 22:
        args += ["-p", str(port)]
    return args


def scp_base_args(port):
    args = ["scp"]
    if int(port) != 22:
        args += ["-P", str(port)]
    return args


def remote_spec(user, host, path):
    return f"{user}@{host}:{path}"


def write_remote_train_script(path: Path):
    path.write_text(r'''
param(
    [Parameter(Mandatory=$true)][string]$JobZip,
    [Parameter(Mandatory=$true)][string]$JobName,
    [Parameter(Mandatory=$true)][string]$ProjectName,
    [Parameter(Mandatory=$true)][string]$BaseModel,
    [Parameter(Mandatory=$true)][int]$ImgWidth,
    [Parameter(Mandatory=$true)][int]$ImgHeight,
    [Parameter(Mandatory=$true)][int]$Epochs,
    [Parameter(Mandatory=$true)][int]$Batch,
    [Parameter(Mandatory=$true)][double]$Lr0,
    [ValidateSet("detect", "classify")][string]$TrainTask = "detect",
    [string]$CondaEnv = "yolov8",
    [string]$TorchCuda = "cu128",
    [ValidateSet("cuda", "cpu")][string]$TrainDevice = "cuda",
    [ValidateSet("False", "True", "disk")][string]$TrainCache = "False"
)


$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch {}
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { [Console]::InputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$JobDir = Join-Path $WorkDir $JobName
$ResultDir = Join-Path $WorkDir "${JobName}_result"
$ResultZip = Join-Path $WorkDir "${JobName}_result.zip"

Remove-Item -Recurse -Force $JobDir, $ResultDir, $ResultZip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $JobDir, $ResultDir | Out-Null
Expand-Archive -Force -Path (Join-Path $WorkDir $JobZip) -DestinationPath $JobDir

if ($TrainTask -eq "detect") {
    $DataYaml = Join-Path $JobDir "dataset.yaml"
    $RemotePath = $JobDir.Replace("\", "/")
    (Get-Content $DataYaml) | ForEach-Object { if ($_ -match '^path:') { "path: $RemotePath" } else { $_ } } | Set-Content -Encoding UTF8 $DataYaml
    $DataArg = "data=$DataYaml"
} else {
    $DataArg = "data=$JobDir"
}

if ([string]::IsNullOrWhiteSpace($CondaEnv)) {
    $PythonCmd = @("python")
    $YoloCmd = @("yolo")
} else {
    $PythonCmd = @("conda", "run", "--no-capture-output", "-n", $CondaEnv, "python")
    $YoloCmd = @("conda", "run", "--no-capture-output", "-n", $CondaEnv, "yolo")
}

function Invoke-Argv($Argv, [object[]]$Rest) {
    $cmd = $Argv[0]
    $args = @()
    if ($Argv.Count -gt 1) { $args += $Argv[1..($Argv.Count - 1)] }
    $args += $Rest
    & $cmd @args
}

function Get-TorchIndexUrl([string]$CudaVersion) {
    $name = $CudaVersion.Trim().ToLower()
    if (@("cu118", "cu121", "cu124", "cu128", "cpu") -contains $name) {
        return "https://download.pytorch.org/whl/$name"
    }
    return ""
}

Invoke-Argv $PythonCmd @("-m", "pip", "install", "-U", "pip")
$TorchIndexUrl = Get-TorchIndexUrl $TorchCuda
if ($TorchIndexUrl) {
    Invoke-Argv $PythonCmd @("-m", "pip", "install", "-U", "torch", "torchvision", "torchaudio", "--index-url", $TorchIndexUrl)
}
Invoke-Argv $PythonCmd @("-m", "pip", "install", "-U", "ultralytics", "onnx", "onnxsim", "onnxruntime-gpu", "pyyaml")

$ImgSize = if ($TrainTask -eq "classify") { [Math]::Max($ImgWidth, $ImgHeight) } else { "$ImgHeight,$ImgWidth" }
$TrainArgv = @(
    $TrainTask, "train",
    "model=$BaseModel",
    $DataArg,
    "imgsz=$ImgSize",
    "epochs=$Epochs",
    "batch=$Batch",
    "lr0=$Lr0",
    "project=$JobDir",
    "name=$ProjectName",
    "pretrained=True",
    "optimizer=AdamW",
    "workers=0",
    "plots=True",
    "cache=$TrainCache",
    "amp=False",
    "device=$TrainDevice"
)
Invoke-Argv $YoloCmd $TrainArgv
$TrainExitCode = $LASTEXITCODE
if ($TrainExitCode -ne 0) { throw "YOLO train failed with exit code $TrainExitCode" }

$BestPt = Join-Path $JobDir "$ProjectName\weights\best.pt"
if (!(Test-Path $BestPt)) { throw "best.pt not found: $BestPt" }

Invoke-Argv $YoloCmd @("export", "model=$BestPt", "format=onnx", "imgsz=$ImgSize", "simplify=True", "opset=17", "dynamic=False")
$BestOnnx = Join-Path $JobDir "$ProjectName\weights\best.onnx"
if (!(Test-Path $BestOnnx)) { throw "best.onnx not found: $BestOnnx" }

Copy-Item $BestPt (Join-Path $ResultDir "best.pt") -Force
Copy-Item $BestOnnx (Join-Path $ResultDir "best.onnx") -Force
$PlotDir = Join-Path $ResultDir "train_plots"
New-Item -ItemType Directory -Force $PlotDir | Out-Null
Get-ChildItem -Path (Join-Path $JobDir $ProjectName) -File -Include *.png,*.jpg,*.jpeg -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName (Join-Path $PlotDir $_.Name) -Force }
Compress-Archive -Force -Path (Join-Path $ResultDir "*") -DestinationPath $ResultZip
Write-Host "REMOTE_RESULT_ZIP=$ResultZip"
'''.lstrip(), encoding="utf-8")


def train_remote_windows(args, yolo_data: Path, work: Path, timestamp: str):
    remote_user_host = f"{args.remote_train_user}@{args.remote_train_host}"
    remote_work_dir = args.remote_train_work_dir.replace("\\", "/").rstrip("/")
    job_name = f"train_job_{timestamp}"
    print(f"REMOTE_JOB_NAME={job_name}", flush=True)
    print(f"REMOTE_WORK_DIR={remote_work_dir}", flush=True)
    dataset_zip = work / f"{job_name}_dataset.zip"
    remote_script = work / f"{job_name}_remote_train.ps1"
    result_zip = work / f"{job_name}_result.zip"
    result_dir = work / f"{job_name}_result"

    zip_dir_contents(yolo_data, dataset_zip)
    write_remote_train_script(remote_script)

    mkdir_cmd = f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"New-Item -ItemType Directory -Force -Path {ps_single_quote(remote_work_dir)} | Out-Null\""
    run(ssh_base_args(args.remote_train_port) + [remote_user_host, mkdir_cmd])
    run(scp_base_args(args.remote_train_port) + [
        str(dataset_zip),
        str(remote_script),
        remote_spec(args.remote_train_user, args.remote_train_host, remote_work_dir + "/"),
    ])

    remote_cmd = (
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        f"{ps_single_quote(remote_work_dir + '/' + remote_script.name)} "
        f"-JobZip {ps_single_quote(dataset_zip.name)} "
        f"-JobName {ps_single_quote(job_name)} "
        f"-ProjectName {ps_single_quote(args.project_name)} "
        f"-BaseModel {ps_single_quote(args.base_model)} "
        f"-ImgWidth {args.img_width} "
        f"-ImgHeight {args.img_height} "
        f"-Epochs {args.epochs} "
        f"-Batch {args.batch} "
        f"-Lr0 {args.lr0} "
        f"-TrainTask {ps_single_quote(args.train_task)} "
        f"-CondaEnv {ps_single_quote(args.conda_env)} "
        f"-TorchCuda {ps_single_quote(args.torch_cuda)} "
        f"-TrainDevice {ps_single_quote(args.train_device)} "
        f"-TrainCache {ps_single_quote(args.train_cache)}"


    )
    print(f"Running remote Windows training on {remote_user_host}:{args.remote_train_port} ...")
    run(ssh_base_args(args.remote_train_port) + [remote_user_host, remote_cmd])
    run(scp_base_args(args.remote_train_port) + [
        remote_spec(args.remote_train_user, args.remote_train_host, remote_work_dir + f"/{job_name}_result.zip"),
        str(result_zip),
    ])

    result_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(result_zip, "r") as zf:
        zf.extractall(result_dir)
    best_pt = result_dir / "best.pt"
    best_onnx = result_dir / "best.onnx"
    if not best_pt.exists() or not best_onnx.exists():
        raise SystemExit(f"Remote training result missing best.pt/best.onnx in {result_dir}")
    return best_pt, best_onnx


def ensure_python_modules(py_cmd, modules: dict[str, str]) -> None:
    missing = []
    for module, package in modules.items():
        result = run(py_cmd + ["-c", f"import {module}"], check=False)
        if result.returncode != 0:
            missing.append(package)
    if missing:
        run(py_cmd + ["-m", "pip", "install"] + missing)



def train_local(args, dataset_path: Path, work: Path):
    py_cmd = build_env_cmd(args.conda_env, "python")
    yolo_cmd = build_env_cmd(args.conda_env, "yolo")
    ensure_python_modules(py_cmd, {
        "torch": "torch",
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
        "ultralytics": "ultralytics",
        "onnx": "onnx",
        "onnxsim": "onnxsim",
        "onnxruntime": "onnxruntime-gpu",
        "yaml": "pyyaml",
    })


    stop_signal = Path(args.stop_export_signal).resolve() if args.stop_export_signal else None
    if stop_signal and stop_signal.exists():
        stop_signal.unlink()
    train_args = [
        args.train_task, "train",
        f"model={args.base_model}",
        f"data={dataset_path}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"lr0={args.lr0}",
        f"project={work}",
        f"name={args.project_name}",
        "pretrained=True",
        "optimizer=AdamW",
        "workers=0",
        "plots=True",
        f"cache={args.train_cache}",
        "amp=False",
        f"device={args.train_device}",
    ]
    if args.train_task == "detect":
        train_args.append(f"imgsz={imgsz_arg(args)}")
    else:
        train_args.append(f"imgsz={max(args.img_width, args.img_height)}")
    train_code = run_train_command(yolo_cmd + train_args, stop_signal=stop_signal)

    best_pt = work / args.project_name / "weights" / "best.pt"
    if train_code != 0 and not best_pt.exists():
        raise subprocess.CalledProcessError(train_code, yolo_cmd + [args.train_task, "train"])
    if not best_pt.exists():
        raise SystemExit(f"best.pt not found: {best_pt}")

    export_size = imgsz_arg(args) if args.train_task == "detect" else max(args.img_width, args.img_height)
    run(yolo_cmd + ["export", f"model={best_pt}", "format=onnx", f"imgsz={export_size}", "simplify=True", "opset=17", "dynamic=False"])
    best_onnx = best_pt.with_suffix(".onnx")
    if not best_onnx.exists():
        raise SystemExit(f"best.onnx not found: {best_onnx}")
    return best_pt, best_onnx



def run_train_stage(args):
    dataset_root = Path(args.dataset_root).resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = dataset_root / f".train_work_{timestamp}"
    yolo_data = work / "yolo_dataset"
    out = dataset_root / f"outputs_{timestamp}"
    work.mkdir(parents=True, exist_ok=True)
    yolo_data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir).resolve() if args.images_dir else dataset_root / "images"
    if args.train_task == "classify":
        dataset_path, _ = prepare_classification_yolo(
            images_dir,
            yolo_data,
            train_ratio_percent=args.train_ratio_percent,
        )
    else:
        annotations_dir = Path(args.annotations_dir).resolve() if args.annotations_dir else dataset_root / "annotations"
        dataset_path, _ = prepare_voc_yolo(
            images_dir,
            annotations_dir,
            yolo_data,
            train_ratio_percent=args.train_ratio_percent,
            img_width=args.img_width,
            img_height=args.img_height,
            resize_mode=args.image_resize_mode,
        )

    best_pt, best_onnx = train_local(args, dataset_path, work)

    model_name = args.model_name or args.project_name
    out_pt = out / f"{model_name}.pt"
    out_onnx = out / f"{model_name}.onnx"
    shutil.copy2(best_pt, out_pt)
    shutil.copy2(best_onnx, out_onnx)
    shutil.copy2(yolo_data / "classes.txt", out / "classes.txt")

    run_dir = best_pt.parent.parent
    plot_dir = out / "train_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for artifact in run_dir.glob(pattern):
            if artifact.is_file():
                shutil.copy2(artifact, plot_dir / artifact.name)

    print(f"TRAIN_OUTPUT_DIR={out}", flush=True)
    print(f"TRAIN_PLOT_DIR={plot_dir}", flush=True)
    print(f"TRAIN_MODEL_PT={out_pt}", flush=True)
    print(f"TRAIN_MODEL_ONNX={out_onnx}", flush=True)
    print(f"TRAIN_CLASSES={out / 'classes.txt'}", flush=True)
    return {
        "timestamp": timestamp,
        "out": out,
        "model_pt": out_pt,
        "model_onnx": out_onnx,
        "classes": out / "classes.txt",
        "model_name": model_name,
    }


def build_parser(script_root: Path):
    ap = argparse.ArgumentParser(description="YOLO training and ONNX export workflow")
    ap.add_argument("--stage", choices=["train"], default="train")
    ap.add_argument("--dataset-root", default=str(script_root))
    ap.add_argument("--images-dir", default="")
    ap.add_argument("--train-task", choices=["detect", "classify"], default="detect", help="训练任务：检测或图像分类")
    ap.add_argument("--annotations-dir", default="")
    ap.add_argument("--train-ratio-percent", type=float, default=80.0, help="训练集占比，1 到 100；验证集占比自动为 100 减该值")


    ap.add_argument("--img-size", type=int, default=None, help="兼容旧调用：同时设置图片宽度和高度")
    ap.add_argument("--img-width", type=int, default=None)
    ap.add_argument("--img-height", type=int, default=None)
    ap.add_argument("--image-resize-mode", choices=["crop", "letterbox", "stretch"], default="letterbox")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--conda-env", default="yolov8")
    ap.add_argument("--base-model", default="yolov8n.pt")
    ap.add_argument("--torch-cuda", choices=["cu118", "cu121", "cu124", "cu128", "cpu", "none"], default="cu128")
    ap.add_argument("--train-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--train-cache", choices=["False", "True", "disk"], default="False")
    ap.add_argument("--stop-export-signal", default="")
    ap.add_argument("--project-name", default="douzi_yolov8n_448")

    ap.add_argument("--model-name", default="douzi_yolov8n_448")
    return ap


def main():
    script_root = Path(__file__).resolve().parent
    args = build_parser(script_root).parse_args()
    validate_image_dimensions(args)
    run_train_stage(args)


if __name__ == "__main__":
    main()
