# model-training-tool

`model-training-tool` 是一个面向 Linux 的 YOLO 浏览器工具。它把数据整理、视频/摄像头跟踪标注、模型训练、ONNX 导出和推理测试集中到一个网页面板中。

## 功能

- 浏览器面板：训练、推理、日志和进度展示。
- 视频、摄像头或图片集跟踪标注，检测框输出 VOC XML，关键点输出 YOLO pose txt。
- 目标检测、图像分类和关键点姿态（YOLO pose）训练。
- 目标检测数据可按视频源分组划分，避免相邻帧同时进入训练集和验证集。
- 训练完成后导出标准 ONNX 与 `.pt` 模型。
- 使用 `.pt` 模型进行摄像头、单图或图片目录推理。
- 检查 Ultralytics YOLO 检测/姿态 ONNX，转换 RK3588/RK3588S FP 或 INT8 RKNN，并生成板端示例包。

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

网页中的“存为默认”也会更新该文件。训练任务使用 `detect`、`classify` 或 `pose`；分类任务要求图片按类别放在子目录中。

### 关键点姿态（pose）

姿态任务读取 YOLO pose txt 标注（不是 VOC XML）。`Annotations Dir` 填标注目录，其中每张图片对应
一个同名 `.txt`，每行一个目标：

```text
cls cx cy w h x0 y0 v0 x1 y1 v1 ... xk yk vk
```

坐标均相对**原图**归一化到 0~1；`v` 为 0（不在图内/未标）、1（遮挡）、2（可见），也可省略 `v`
（视为可见）。类别名取 `--pose-classes`，留空时读取标注目录下的 `classes.txt`。
可用 CVAT、Label Studio 等导出 “YOLO Pose / Ultralytics” 格式。
也可直接用本工具“视频打标”页的网页工作台标注：标注任务选“关键点 Pose”，填写关键点数量和按顺序排列的名称，
开始标注后点“添加关键点目标”，按顺序点击各点（Shift+点击为遮挡 v=1，`S` 跳过为 v=0，`Backspace` 撤销，`Esc` 取消）。
框由关键点外扩自动生成，也可“修正选中框”手动重画；空闲时可拖动已有关键点微调。标注目录会写出同名 `.txt` 和 `classes.txt`，
可直接作为训练页的 Pose Labels Dir。播放跟踪时关键点只随框平移缩放，属于近似结果，保存前需检查修正。

工具会按所选缩放方式（crop/letterbox/stretch）同步变换关键点，落到输出图外的点自动置为 `v=0`，
并生成带 `kpt_shape`、`flip_idx`（和可选 `kpt_oks_sigmas`）的 `dataset.yaml`。
`flip_idx` 描述左右翻转后关键点的对应关系；留空时 Ultralytics 会关闭翻转增强。
基础权重请使用姿态模型，例如 `yolov8n-pose.pt`。

```bash
python host_train_export.py --train-task pose \
  --images-dir /path/images --annotations-dir /path/labels \
  --kpt-count 6 --flip-idx 1,0,3,2,4,5 \
  --base-model yolov8n-pose.pt --img-width 448 --img-height 448
```

导出的 ONNX 输出为 `[1, 4+类别数+关键点数*3, N]`。

## 启动网页面板

```bash
.venv/bin/python train_panel.py --host 127.0.0.1 --port 8989
```

用哪个 Python 启动面板，训练就用哪个环境：`yolo` 命令会优先从该 Python 所在目录（如 `.venv/bin/yolo`）查找，因此不需要先 `activate`。训练日志开头的 `YOLO_CMD=` 一行显示实际使用的 `yolo` 路径。

浏览器访问 <http://127.0.0.1:8989>。需要局域网访问时，将 `--host` 改为 `0.0.0.0`，并使用本机 IP 访问。

页面代码、HTTP 接口和网页标注工作台都内嵌在 [`train_panel.py`](train_panel.py) 中。训练、推理和标注的命令行功能分别由以下模块实现：

- [`host_train_export.py`](host_train_export.py)：数据集准备、训练和 ONNX 导出。
- [`video_track_label.py`](video_track_label.py)：视频、摄像头和图片集标注。
- [`model_test.py`](model_test.py)：模型推理测试。
- [`rk3588_deploy.py`](rk3588_deploy.py)：RKNN 环境/ONNX 检查、模型转换和部署包生成。
- [`record_camera.py`](record_camera.py)：USB 摄像头录像，采集训练视频。

## 完整训练流程（网页面板）

先用 `record_camera.py` 录视频，再按顺序使用面板的 5 个页签：

```text
record_camera.py → 04 视频打标 → 01 训练配置 → 02 模型测试 → 03 RK3588 转换 → 开发板运行
  （录制视频）     （标注）     （训练+导出ONNX）（.pt 效果检查） （RKNN + 部署包）
                         05 运行日志：所有任务的实时输出、停止任务
```

三种模型的差别：

| 任务 | 用途 | 标注格式 | 推荐 Base Model | RK3588 转换 |
| --- | --- | --- | --- | --- |
| 目标检测 `detect` | 找出目标并画框 | 图片 + 同名 VOC XML | `yolov8n.pt` | 面板直接转换，推荐 INT8 |
| 关键点姿态 `pose` | 框 + 每个目标的若干关键点（如安全区角点） | 图片 + 同名 YOLO pose txt + `classes.txt` | `yolov8n-pose.pt` | 面板直接转换，推荐 FP |
| 图像分类 `classify` | 整张图属于哪一类 | `Images Dir/类别名/图片`，不需要标注文件 | `yolov8n-cls.pt` 或 `yolo11n-cls.pt` | 面板不支持，见下文手动转换 |

### 第 1 步：采集视频

推荐先录成视频文件再标注：视频可以反复回看、重标，比摄像头边采边标更稳妥。

#### 用 record_camera.py 录制

[`record_camera.py`](record_camera.py) 通过 V4L2 打开 USB 摄像头（MJPG），录成 mp4：

```bash
.venv/bin/python record_camera.py --camera 2 --width 1280 --height 720 --fps 30 --output-dir videos --prefix zone
```

- 空格开始/停止录一段，`q` 或 `Esc` 退出；每段保存为 `videos/zone_年月日_时分秒.mp4`。
- 启动时打印摄像头实际的分辨率和帧率；摄像头不支持所设分辨率时会自动降档，以打印值为准。
- `--camera N` 对应 `/dev/videoN`；用 `ls /dev/video*` 查看可用设备。打不开时检查是否被其他程序占用。
- `--duration 60`：每段最长 60 秒，到时自动停止。
- `--no-preview`：不开窗口、启动即录，按 `Ctrl+C` 或到时长结束，适合在机器人上通过 SSH 采集。

其他设备录的视频（mp4、avi、mov、mkv、webm 等）也可以直接使用，放进同一个文件夹即可。

#### 采集要求

1. **用最终部署的相机采**：同型号、同分辨率、同安装高度和俯角，最好装在机器人上边走边录。换相机或视角后精度会明显下降。
2. **分段录制，段数比时长重要**：每段 30 秒～2 分钟，一段一种场景，总数 10 段以上。训练时按视频分组划分训练/验证集，段数太少验证集只有一两段，指标没有参考意义。
3. **覆盖多样性**：
   - 距离：远、中、近，包括目标很小和几乎占满画面。
   - 角度：正对、斜对、左右偏转。
   - 光照：正常、偏暗、逆光、阴影、反光。
   - 干扰：颜色相近的物体、地面标线、人或其他机器人入镜。
   - 遮挡：目标部分被挡或部分出画。
4. **运动放慢**：匀速慢移，避免严重运动模糊；保留少量轻微模糊片段，贴近真实运行。
5. **画面干净**：关闭摄像头时间水印/OSD。自动曝光在明暗切换时变化剧烈，可固定曝光，或多录一些明暗切换过程。
6. **Pose（安全区关键点）额外要求**：
   - 两个安全区紧挨着时的各种相对位置都要录到，这是 pose 要解决的难点。
   - 录一些角点被遮挡或出画的片段，标注时标为遮挡（v=1）或跳过（v=0）。
   - 大部分片段中分隔线两端应清晰可见。
7. **数据量估算**：30 fps 视频、`Save Every N Frames=10` 时约每秒 3 张，1 分钟约 180 张。一个模型一般需要 1000～3000 张标注图，每个类别几百个目标实例。

录完后，在“视频打标”页的 `Video Folder` 填 `videos` 所在的绝对路径，点“读取文件夹视频”。选中视频时 `Filename Prefix` 会按视频名自动填好（手动改过则不再自动填），保证每段视频的帧前缀不同。

### 第 2 步：标注（视频打标）

1. **标注来源**：选“视频 / 摄像头 / 图片集”。
   - 视频：填 `Video Folder` → “读取文件夹视频” → 在视频队列里选一个（可用“上一个/下一个”切换）。
   - 摄像头：填 `Camera Index`（通常 0）。
   - 图片集：选已按时间顺序命名的帧图片文件夹；原图不复制、不修改，只在 `Annotations Dir` 生成同名标注。
2. **标注任务**：
   - 检测框（VOC XML）：用于 `detect`。
   - 关键点 Pose（YOLO pose txt）：用于 `pose`。再填 `Keypoints`（关键点数量）和 `Keypoint Names`（按点击顺序，逗号分隔）。
3. **参数**：
   - `Labels`：类别名，逗号分隔，例如 `zone_red,zone_blue`。Pose 模式下类别顺序即类别 id，写入 `classes.txt` 后不要再改顺序。
   - `Filename Prefix`：导出帧名前缀，每段视频用不同前缀（训练时按前缀分组划分，防止同一视频同时进入训练集和验证集）。
   - `Images Dir` / `Annotations Dir`：导出图片与标注目录。
   - `Tracker`：一般用 CSRT；目标角度变化大时用 Multi-template。
   - `Save Every N Frames`：每 N 帧自动保存一次，相邻帧太像时调大（5~15）。
4. 点“在网页中开始标注”，在下方“网页标注工作台”里操作：
   - 检测：点“添加框”，在画面上拖出框并选类别；可添加多个目标。
   - Pose：点“添加关键点目标”，按名称顺序依次点击每个关键点（Shift+点击=遮挡 v=1，`S`=看不到跳过 v=0，`Backspace` 撤销，`Esc` 取消），框自动生成。
   - 点“播放跟踪”让目标自动跟随；跟丢会自动暂停，用“修正选中框”重画或“删除选中框”。Pose 的关键点只随框近似移动，偏了就暂停后拖动关键点修正或“重标选中目标关键点”。
   - “保存当前帧”可手动保存；“结束标注”关闭会话。
5. 右侧“标注结果”可预览已保存样本，删除废图。也可以“复制当前命令”，在桌面环境中用 [`video_track_label.py`](video_track_label.py) 的 OpenCV 窗口标注（见“常用命令”）。

分类数据不需要这个页面：直接把图片按类别放进子文件夹，例如 `images/normal/*.jpg`、`images/defect/*.jpg`，每类至少 2 张（建议每类几百张）。

数据量建议：每个类别至少几百个目标实例，覆盖不同光照、距离、角度和背景；最终部署在什么相机上，就尽量用那台相机采集。

### 第 3 步：训练（训练配置）

1. **训练任务**：选目标检测 / 图像分类 / 关键点姿态。
2. **路径**：
   - `Dataset Root / Output Root`：工作与输出根目录。训练会在这里生成 `.train_work_时间戳/`（临时 YOLO 数据集和训练过程）和 `outputs_时间戳/`（最终产物）。
   - `Images Dir`：检测/Pose 填图片目录；分类填类别子文件夹的上级目录。
   - `Annotations Dir`：检测填 XML 目录；Pose 时标题变为 `Pose Labels Dir`，填 txt 目录；分类会隐藏。
3. **Pose 专用**：
   - `关键点数量`：与标注一致，例如 6。
   - `flip_idx`：左右翻转后关键点的对应关系。安全区 6 点顺序为 `entry_left,entry_right,inner_right,inner_left,divider_entry,divider_inner`，填 `1,0,3,2,4,5`；关键点没有左右对称关系时留空（关闭翻转增强）。
   - `类别名称`：留空时读取标注目录的 `classes.txt`。
   - `OKS sigmas`：一般留空。
4. **划分**：比例建议 80/20；视频帧数据选“按视频源分组（推荐）”，独立拍摄的图片才选“逐图片随机”。
5. **模型与环境**：
   - `Base Model`：检测 `yolov8n.pt`，Pose `yolov8n-pose.pt`，分类 `yolov8n-cls.pt`。首次训练会自动下载。需要更高精度时换 `s` 版本（如 `yolov8s-pose.pt`），但 RK3588 上推理会明显变慢，需实测帧率。
   - `Conda Env`：填了且本机有 conda 时，用 `conda run -n 环境名` 运行训练，该环境必须已存在。本机没有 conda 时会忽略该项（日志提示 `info=未找到 conda…`），直接使用启动面板的 Python（例如 `.venv/bin/python`）。缺少 ultralytics/onnx 等模块时会自动 `pip install` 到该环境。
   - `PyTorch CUDA`：本机训练不会按此选项重装 PyTorch，缺 torch 时只会装默认版本。请事先在训练环境装好带 CUDA 的 PyTorch，并确认下面命令输出 `True`：

     ```bash
     .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
     ```

     输出 `False` 时按显卡驱动重装，例如 CUDA 12.8：

     ```bash
     .venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```
   - `Train Device`：有 NVIDIA 显卡选 CUDA/GPU。
   - `Cache`：内存充足时选“开启 cache 到内存”可加速。
   - `Project Name` / `Model Name`：输出文件名，例如 `safe_zone_pose`。
6. **图片尺寸**：
   - `图片宽度` / `图片高度`：必须是 32 的倍数，这也是之后 ONNX/RKNN 的固定输入尺寸。RK3588 推荐 `448×448`（速度与精度折中）或 `640×640`（小目标更好）。分类任务取两者中的较大值。
   - `图片适配方式`：推荐“等比缩放（留边填充）”，与板端 `infer_image.py` 的 letterbox 预处理一致。
7. **超参数**：`Epochs` 检测/Pose 建议 100~300、分类 50~100；`Batch` 8 GB 显存在 448 下 n 模型可用 16~32，显存不足就调小；`Lr0` 默认即可。
8. 右侧“资源预估”会估算内存和显存。点“开始训练”，或“复制训练命令”在终端运行。

训练中页面显示进度、损失曲线和验证指标：检测看 `mAP50-95`，Pose 同时看 `Pose mAP50-95`（关键点精度），分类看 `Top-1 Accuracy`。“运行日志”页可查看完整输出和“停止任务”。

训练结束后，`Dataset Root/outputs_时间戳/` 中有：

- `模型名.pt`：PyTorch 权重，用于模型测试和再次导出。
- `模型名.onnx`：静态 ONNX（batch=1、固定尺寸、simplify、opset 17），用于 RK3588 转换。
- `classes.txt`：类别文件。
- `train_plots/`：训练曲线、混淆矩阵、验证样例图。

面板会把 ONNX 和 `classes.txt` 自动填入“RK3588 转换”页（该页原本为空时）。

### 第 4 步：测试（模型测试）

`Test Model .pt` 选上一步的 `.pt`，`Source` 选摄像头、单张图片或图片文件夹，点“开始测试”。检测、Pose、分类模型自动识别；单图保存带结果的预测图，文件夹模式递归处理并输出 `predictions.csv`。`Confidence` 仅对检测/Pose 生效。在电脑上效果不好时，先补数据重训，不要急着转换。

### 第 5 步：转换为 RK3588 模型（RK3588 转换）

先准备一个独立的 RKNN 转换环境（不要装进本项目 `.venv`）：

```bash
python3 -m venv ~/rknn-env
~/rknn-env/bin/python -m pip install /path/to/rknn_toolkit2-<版本>-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
~/rknn-env/bin/python -m pip install onnx opencv-python numpy
```

wheel 来自板卡厂商 SDK 或 Rockchip `rknn-toolkit2` 仓库，版本要和板端 `librknnrt.so`/RKNPU 驱动匹配。

页面操作：

1. `ONNX 模型`、`类别文件`：训练后已自动填好，否则选 `outputs_时间戳/` 下的 `.onnx` 与 `classes.txt`。
2. `RKNN Python`：填 `~/rknn-env/bin/python` 的绝对路径，点“检查环境”。
3. 点“检查模型”：会检查单输入、batch=1、固定尺寸、宽高为 32 的倍数，以及输出通道数与类别数/关键点数是否匹配。
4. `转换模式`：
   - 检测模型：选“先 FP，再 INT8（推荐）”，两个都生成，板上对比后用 INT8。
   - Pose 模型：选“仅非量化 FP”。关键点坐标对 INT8 量化误差敏感，FP（NPU 上为 FP16）更稳定。
5. `校准图片目录`（INT8 必填）：放 100~200 张部署现场的真实图片（可直接用训练图片目录），至少 20 张；`校准图片数量` 默认 200。
6. `测试图片`（可选）：转换后在 PC 模拟器上跑一次，确认输出形状正常。
7. `置信度阈值` / `NMS IoU`：写入部署包默认值，一般 0.25 / 0.45。
8. 点“开始转换并打包”，完成后点“下载部署包”。

输出目录（默认在 ONNX 同目录下的时间戳目录）包含 `model_fp.rknn`、`model_int8.rknn`（按模式）、转换清单和 `rk3588_deploy.zip`。

### 第 6 步：在开发板运行

把 `rk3588_deploy.zip` 拷到 RK3588 板上解压，安装厂商 SDK 提供的 `rknn_toolkit_lite2` wheel 和 `numpy`、`opencv-python`，然后运行：

```bash
cd rk3588_yolo
python3 infer_image.py --model models/model_fp.rknn --image test.jpg --output result.jpg
```

INT8 模型把 `--model` 换成 `models/model_int8.rknn`。Pose 模型会额外打印每个关键点的像素坐标和可见性，`--kpt-conf` 控制绘制阈值。板端输入为 letterbox 后的 RGB uint8，不要再除以 255（归一化已写进模型）。

### 最适合 RK3588 的配置建议

- **模型大小**：优先 `n` 版本，精度不够再换 `s`。RK3588 NPU 为 3 核共 6 TOPS，`m` 及以上一般难以实时。
- **模型系列**：`yolov8` 系列在 RKNN-Toolkit2 上兼容性最好；`yolo11` 含注意力结构，也能转换，但部分算子可能回退 CPU，速度需实测。
- **输入尺寸**：固定、32 的倍数，训练、导出、转换、推理四处一致。448 或 640 最常用；在满足精度的前提下越小越快。
- **ONNX 形式**：静态 shape、batch=1、不带 NMS（本工具默认导出即是）。若 Toolkit 报 opset 或算子不支持，用 `.pt` 重新导出 opset 12，无需重训：

  ```bash
  yolo export model=outputs_时间戳/模型名.pt format=onnx imgsz=448,448 opset=12 simplify=True dynamic=False
  ```

- **量化**：检测用 INT8（通常明显快于 FP，精度损失一般较小，以板上实测为准），校准图片必须来自真实场景；Pose 优先 FP，如需 INT8 务必在板上对比关键点误差。分类 INT8 一般可用。
- **验证**：转换后先用“测试图片”在模拟器确认输出形状，再到板上用同一张图对比 FP 与 INT8 的结果。

### 分类模型转换（手动）

`rk3588_deploy.py` 只支持检测和 Pose 的输出格式，分类模型需要在 RKNN 环境中手动转换。训练得到的 `.onnx` 为静态 `[1,3,S,S]` 输入、`[1,类别数]` 概率输出：

```python
# convert_cls.py：用 ~/rknn-env/bin/python convert_cls.py 运行
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
assert rknn.load_onnx(model="outputs_时间戳/模型名.onnx") == 0
# FP：do_quantization=False；INT8：do_quantization=True, dataset="calib.txt"（每行一张图片的绝对路径）
assert rknn.build(do_quantization=False) == 0
assert rknn.export_rknn("model_cls_fp.rknn") == 0
rknn.release()
```

板端推理时把图片缩放到 `S×S`（与训练一致，Ultralytics 分类默认中心裁剪后缩放）、转 RGB，以 uint8 NHWC 输入，输出即各类别概率，取最大值对应 `classes.txt` 中的类别。该流程未经本工具自动验证，请在板上用几张已知类别图片确认。

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

`--mode` 支持 `fp`、`int8` 和 `both`。姿态模型会从 ONNX 元数据自动识别 `task=pose` 与
`kpt_shape`（也可用 `--kpt-count` 手动指定），板端 `infer_image.py` 会解码并绘制关键点；
关键点精度对量化敏感，建议优先使用 FP 模型。INT8 至少需要 20 张校准图片，建议使用 100 张以上
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
python record_camera.py --help
```

直接标注视频的示例：

```bash
python video_track_label.py \
  --video /home/user/Videos/input.mp4 \
  --labels object \
  --images-dir /home/user/datasets/helmet/images \
  --annotations-dir /home/user/datasets/helmet/annotations
```

Pose 关键点标注（输出 YOLO pose txt + `classes.txt`，可直接交给 pose 数据准备）：

```bash
python video_track_label.py \
  --video /home/user/Videos/safe_zone.mp4 \
  --labels zone_red,zone_blue \
  --task pose --kpt-count 6 \
  --kpt-names entry_left,entry_right,inner_right,inner_left,divider_entry,divider_inner \
  --images-dir /home/user/datasets/zone/images \
  --annotations-dir /home/user/datasets/zone/labels
```

按 `a` 后按顺序单击关键点（Shift+单击=遮挡，`s` 跳过，`u`/退格撤销，回车确认），再选类别；框由关键点自动生成，`r` 可手动改框。跟踪时关键点随框近似移动，偏了就暂停按 `k` 重标。

## 验证

提交前运行：

```bash
python -m py_compile train_panel.py host_train_export.py model_test.py video_track_label.py rk3588_deploy.py record_camera.py
python train_panel.py --help
python host_train_export.py --help
python model_test.py --help
```

不要提交默认配置、数据集、视频、标注、模型权重、压缩包或训练输出目录。
