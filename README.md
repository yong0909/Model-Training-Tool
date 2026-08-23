# model-training-tool

`model-training-tool` 是一个面向 Linux 的 YOLO 浏览器工具。它把数据整理、视频/摄像头跟踪标注、模型训练、ONNX 导出和推理测试集中到一个网页面板中。

## 功能

- 浏览器面板：训练、推理、日志和进度展示。
- 视频、摄像头或图片集跟踪标注，输出 VOC XML。
- 目标检测和图像分类训练。
- 目标检测数据可按视频源分组划分，避免相邻帧同时进入训练集和验证集。
- 训练完成后导出标准 ONNX 与 `.pt` 模型。
- 使用 `.pt` 模型进行摄像头、单图或图片目录推理。
- 检查 Ultralytics YOLO 检测 ONNX，转换 RK3588/RK3588S FP 或 INT8 RKNN，并生成板端示例包。

## 环境

- Linux（推荐 Ubuntu 22.04 或更新版本）
- Python 3.10、3.11 或 3.12
- 摄像头标注需要可用的 V4L2 设备和图形桌面；服务器环境可使用视频或图片集模式。

安装依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果网页中的文件夹选择按钮不可用，请安装 Tk：

```bash
sudo apt install python3-tk
```

## 配置

复制默认配置示例：

```bash
cp train_panel_defaults.example.json train_panel_defaults.json
```

`train_panel_defaults.json` 只保存本机设置，可能包含本地目录，不要提交到 Git。Linux 路径使用绝对路径，例如：

```json
{
  "dataset_root": "/home/user/datasets/helmet",
  "train_images_dir": "/home/user/datasets/helmet/images",
  "train_annotations_dir": "/home/user/datasets/helmet/annotations",
  "split_mode": "video_group",
  "train_task": "detect",
  "train_device": "cuda",
  "base_model": "yolov8n.pt",
  "label_video_dir": "/home/user/Videos/raw",
  "label_images_dir": "/home/user/datasets/helmet/images",
  "label_annotations_dir": "/home/user/datasets/helmet/annotations"
}
```

目标检测默认使用 `video_group` 划分：网页标注产生的图片名形如
`视频前缀_000015.jpg`，工具会移除末尾帧号作为视频源标识，并将同一视频的
全部帧放入同一个集合。训练/验证比例会在整段视频的粒度上尽量接近设定值，
同时兼顾各类别数量；因此实际图片比例可能不是严格的 80%/20%。只有图片之间
本来就相互独立时，才选择 `random` 逐图片随机划分。每次整理后的 YOLO 数据集
还会生成 `split_manifest.csv`，记录每张图片的集合和视频源，可用于检查两边是否
存在同源视频。

网页中的“存为默认”也会更新该文件。训练任务使用 `detect` 或 `classify`；分类任务要求图片按类别放在子目录中。

## 启动网页面板

```bash
python train_panel.py --host 127.0.0.1 --port 8989
```

浏览器访问 <http://127.0.0.1:8989>。需要局域网访问时，将 `--host` 改为 `0.0.0.0`，并使用本机 IP 访问。

页面代码、HTTP 接口和网页标注工作台都内嵌在 [`train_panel.py`](train_panel.py) 中。训练、推理和标注的命令行功能分别由以下模块实现：

- [`host_train_export.py`](host_train_export.py)：数据集准备、训练和 ONNX 导出。
- [`video_track_label.py`](video_track_label.py)：视频、摄像头和图片集标注。
- [`model_test.py`](model_test.py)：模型推理测试。
- [`rk3588_deploy.py`](rk3588_deploy.py)：RKNN 环境/ONNX 检查、模型转换和部署包生成。

## RK3588 转换与打包

网页中的“RK3588 转换”页签在独立 Python 环境中调用 RKNN-Toolkit2。项目不会自动安装
`rknn-toolkit2`，也不会把它加入根目录 `requirements.txt`：请从板卡厂商 SDK 获取与板端
RKNN Runtime/RKNPU 驱动匹配的 wheel，并把该虚拟环境的 Python 路径填入网页。转换功能不保存
SSH 信息，也不自动上传开发板。

命令行示例：

```bash
/home/user/rknn-env/bin/python rk3588_deploy.py \
  --onnx /path/model.onnx \
  --classes /path/classes.txt \
  --output-dir /path/rk3588_deploy \
  --mode both \
  --calibration-dir /path/calibration_images \
  --calibration-count 200 \
  --test-image /path/test.jpg \
  --conf 0.25 \
  --iou 0.45
```

`--mode` 支持 `fp`、`int8` 和 `both`。INT8 至少需要 20 张校准图片，建议使用 100 张以上
有代表性的真实图片。若 Toolkit 提示 opset 或算子不支持，可从原 `.pt` 重新导出静态
`opset=12` ONNX，无需重新训练。

生成的 ZIP 包含 RKNN 模型、类别、转换清单、校准列表和可在开发板运行的
`infer_image.py`。包内不包含 Lite2 wheel、`librknnrt.so` 或驱动文件，这些文件必须来自
板卡厂商配套 SDK。在真实 RK3588/RK3588S 上验证前，此功能仅表示转换与打包完成。

## 常用命令

```bash
python host_train_export.py --help
python video_track_label.py --help
python model_test.py --help
python rk3588_deploy.py --help
```

直接标注视频的示例：

```bash
python video_track_label.py \
  --video /home/user/Videos/input.mp4 \
  --labels object \
  --images-dir /home/user/datasets/helmet/images \
  --annotations-dir /home/user/datasets/helmet/annotations
```

## 验证

提交前运行：

```bash
python -m py_compile train_panel.py host_train_export.py model_test.py video_track_label.py rk3588_deploy.py
python train_panel.py --help
python host_train_export.py --help
python model_test.py --help
```

不要提交默认配置、数据集、视频、标注、模型权重、压缩包或训练输出目录。
