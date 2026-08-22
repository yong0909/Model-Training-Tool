#!/usr/bin/env bash
set -euo pipefail

# 该脚本会被 host_train_export.py 复制到每个转换任务包中。
# 历史 outputs_*、convert_outputs_* 中的副本不会随根脚本更新。
CONVERT_SCRIPT_REVISION="20260726-yolov8-head-outputs-v2"

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MUD_MODEL_TYPE="${MUD_MODEL_TYPE:-yolov8}"
MUD_ANCHORS="${MUD_ANCHORS:-}"
IMAGE_NAME="${IMAGE_NAME:-sophgo/tpuc_dev:latest}"
AUTO_INSTALL_TPU_MLIR="${AUTO_INSTALL_TPU_MLIR:-1}"
TPU_MLIR_PIP_SPEC="${TPU_MLIR_PIP_SPEC:-tpu_mlir}"

echo "CONVERT_SCRIPT_REVISION=${CONVERT_SCRIPT_REVISION}"
echo "IMAGE_NAME=${IMAGE_NAME}"
echo "AUTO_INSTALL_TPU_MLIR=${AUTO_INSTALL_TPU_MLIR}"

mapfile -t ONNX_FILES < <(find "$JOB_DIR" -maxdepth 1 -type f -iname '*.onnx' -print)
if [ "${#ONNX_FILES[@]}" -ne 1 ]; then
  echo "expected exactly one ONNX file in: $JOB_DIR" >&2
  printf 'found: %s\n' "${ONNX_FILES[@]:-none}" >&2
  exit 1
fi

ONNX_PATH="${ONNX_FILES[0]}"
ONNX="$(basename "$ONNX_PATH")"
MODEL_NAME="${ONNX%.onnx}"
OUT_DIR="$JOB_DIR/outputs_${TS}"

if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid model name: $MODEL_NAME" >&2
  exit 1
fi
if [[ ! "$MUD_MODEL_TYPE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid MUD_MODEL_TYPE: $MUD_MODEL_TYPE" >&2
  exit 1
fi
MUD_MODEL_TYPE="${MUD_MODEL_TYPE,,}"
if [[ -n "$MUD_ANCHORS" && ! "$MUD_ANCHORS" =~ ^[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*$ ]]; then
  echo "MUD_ANCHORS must be comma-separated non-negative integers" >&2
  exit 1
fi
if [[ "${MUD_MODEL_TYPE,,}" == "yolov5" ]]; then
  if [[ -z "$MUD_ANCHORS" ]]; then
    echo "MUD_ANCHORS is required when MUD_MODEL_TYPE=yolov5" >&2
    exit 1
  fi
  IFS=',' read -r -a ANCHOR_VALUES <<< "$MUD_ANCHORS"
  if (( ${#ANCHOR_VALUES[@]} % 2 != 0 )); then
    echo "MUD_ANCHORS must contain width,height pairs" >&2
    exit 1
  fi
fi

for required_file in "$JOB_DIR/classes.txt" "$JOB_DIR/test.jpg"; do
  if [[ ! -f "$required_file" ]]; then
    echo "required conversion input is missing: $required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$JOB_DIR/calib_images" ]]; then
  echo "required calibration directory is missing: $JOB_DIR/calib_images" >&2
  exit 1
fi
if ! find "$JOB_DIR/calib_images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit | grep -q .; then
  echo "no JPG/JPEG/PNG calibration images found: $JOB_DIR/calib_images" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cp "$ONNX_PATH" "$OUT_DIR/$ONNX"
cp "$JOB_DIR/classes.txt" "$OUT_DIR/"
cp "$JOB_DIR/test.jpg" "$OUT_DIR/"
rm -rf "$OUT_DIR/calib_images"
cp -r "$JOB_DIR/calib_images" "$OUT_DIR/calib_images"

cat > "$OUT_DIR/convert_inside_container.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
# `/workspace` 是 Sophgo tpuc_dev 镜像中 TPU-MLIR 源码/环境的常见位置，
# 不能将转换文件挂载到该目录，否则会遮蔽镜像内的工具链。
cd /work

ONNX_FILES=(./*.onnx)
if [ "${#ONNX_FILES[@]}" -ne 1 ] || [ ! -f "${ONNX_FILES[0]}" ]; then
  echo "expected exactly one ONNX file in /work" >&2
  exit 1
fi
ONNX="${ONNX_FILES[0]#./}"
MODEL_NAME="${ONNX%.onnx}"
read -r IMG_HEIGHT IMG_WIDTH < <(python3 - "$ONNX" <<'PY'
import sys

import onnx

model = onnx.load(sys.argv[1])
shape = model.graph.input[0].type.tensor_type.shape.dim
if len(shape) != 4 or not shape[2].dim_value or not shape[3].dim_value:
    raise SystemExit("ONNX input must have static NCHW dimensions")
print(shape[2].dim_value, shape[3].dim_value)
PY
)
CVIMODEL="${MODEL_NAME}_int8.cvimodel"
echo "Model: ${ONNX} (${IMG_WIDTH}x${IMG_HEIGHT})"

if [[ -f /workspace/tpu-mlir/envsetup.sh ]]; then
  # Sophgo tpuc_dev 的已构建 TPU-MLIR 环境通常在此路径。
  source /workspace/tpu-mlir/envsetup.sh
elif [[ -f /workspace/envsetup.sh ]]; then
  source /workspace/envsetup.sh
elif [[ -f /opt/tpu-mlir/envsetup.sh ]]; then
  source /opt/tpu-mlir/envsetup.sh
fi

has_tpu_mlir_tools() {
  command -v model_transform.py >/dev/null 2>&1 && \
    command -v model_deploy.py >/dev/null 2>&1 && \
    command -v run_calibration.py >/dev/null 2>&1
}

if ! has_tpu_mlir_tools && [[ "$AUTO_INSTALL_TPU_MLIR" == "1" ]]; then
  echo "TPU-MLIR tools not found; installing ${TPU_MLIR_PIP_SPEC} in the current container..."
  python3 -m pip install --no-input --disable-pip-version-check "$TPU_MLIR_PIP_SPEC"
  hash -r
fi

if ! has_tpu_mlir_tools; then
  echo "TPU-MLIR tools are unavailable after environment initialization and optional installation." >&2
  echo "Checked /workspace/tpu-mlir/envsetup.sh, /workspace/envsetup.sh, and /opt/tpu-mlir/envsetup.sh." >&2
  echo "Set AUTO_INSTALL_TPU_MLIR=1 with a reachable package source, or use a verified image containing the tools." >&2
  exit 1
fi
printf 'TPU-MLIR tools: transform=%s deploy=%s calibration=%s\n' \
  "$(command -v model_transform.py)" \
  "$(command -v model_deploy.py)" \
  "$(command -v run_calibration.py)"

OUTPUT_NAMES="$(python3 - "$ONNX" "$MUD_MODEL_TYPE" <<'PY'
import sys

import onnx

onnx_path, model_type = sys.argv[1:]
m = onnx.load(onnx_path)
graph_outputs = [output.name for output in m.graph.output]
if not graph_outputs:
    raise SystemExit("ONNX graph has no declared outputs")

if model_type == "yolov8":
    node_outputs = {name for node in m.graph.node for name in node.output}
    dfl_candidates = [
        name for name in node_outputs
        if name.endswith("/dfl/conv/Conv_output_0")
    ]
    # 检测头的分类 Sigmoid 位于末级 `model.<head>/Sigmoid`；骨干网络中的
    # `.../act/Sigmoid` 是 SiLU 激活的一部分，不能作为模型输出。
    sigmoid_candidates = [
        name for name in node_outputs
        if name.endswith("/Sigmoid_output_0") and "/act/" not in name
    ]
    if len(dfl_candidates) != 1 or len(sigmoid_candidates) != 1:
        raise SystemExit(
            "cannot locate exactly one YOLOv8 DFL and classification Sigmoid output; "
            f"dfl={dfl_candidates}, sigmoid={sigmoid_candidates}"
        )
    print(f"{dfl_candidates[0]},{sigmoid_candidates[0]}")
else:
    print(",".join(graph_outputs))
PY
)"
echo "TPU-MLIR output names: ${OUTPUT_NAMES}"

model_transform.py \
  --model_name "$MODEL_NAME" \
  --model_def "./$ONNX" \
  --input_shapes "[[1,3,${IMG_HEIGHT},${IMG_WIDTH}]]" \
  --mean "0,0,0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --channel_format nchw \
  --output_names "$OUTPUT_NAMES" \
  --test_input ./test.jpg \
  --test_result "${MODEL_NAME}_top_outputs.npz" \
  --tolerance 0.99,0.99 \
  --mlir "${MODEL_NAME}.mlir"

CALIB_NUM=$(find ./calib_images -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 200 | wc -l)
if [ "$CALIB_NUM" -lt 1 ]; then
  echo "no calibration images found"
  exit 1
fi

run_calibration.py "${MODEL_NAME}.mlir" \
  --dataset ./calib_images \
  --input_num "$CALIB_NUM" \
  -o "${MODEL_NAME}_cali_table"

model_deploy.py \
  --mlir "${MODEL_NAME}.mlir" \
  --quantize INT8 \
  --quant_input \
  --calibration_table "${MODEL_NAME}_cali_table" \
  --processor cv181x \
  --test_input "${MODEL_NAME}_in_f32.npz" \
  --test_reference "${MODEL_NAME}_top_outputs.npz" \
  --tolerance 0.9,0.6 \
  --model "$CVIMODEL"
EOS
chmod +x "$OUT_DIR/convert_inside_container.sh"

docker run --rm --privileged \
  -v "$OUT_DIR:/work" \
  -e "AUTO_INSTALL_TPU_MLIR=${AUTO_INSTALL_TPU_MLIR}" \
  -e "TPU_MLIR_PIP_SPEC=${TPU_MLIR_PIP_SPEC}" \
  -e "MUD_MODEL_TYPE=${MUD_MODEL_TYPE}" \
  -e "MUD_ANCHORS=${MUD_ANCHORS}" \
  "$IMAGE_NAME" \
  bash /work/convert_inside_container.sh

LABELS=$(paste -sd ', ' "$OUT_DIR/classes.txt")
{
  printf '%s\n' '[basic]' 'type = cvimodel' "model = ${MODEL_NAME}_int8.cvimodel" '' '[extra]'
  printf '%s\n' "model_type = ${MUD_MODEL_TYPE}" 'input_type = rgb' 'mean = 0, 0, 0' 'scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098'
  if [[ -n "$MUD_ANCHORS" ]]; then
    printf '%s\n' "anchors = ${MUD_ANCHORS}"
  fi
  printf '%s\n' "labels = ${LABELS}"
} > "$OUT_DIR/${MODEL_NAME}.mud"

FINAL_DIR="$OUT_DIR"
tar -czf "${FINAL_DIR}.tar.gz" -C "$(dirname "$FINAL_DIR")" "$(basename "$FINAL_DIR")"
echo "Done: $FINAL_DIR"
echo "Archive: ${FINAL_DIR}.tar.gz"
