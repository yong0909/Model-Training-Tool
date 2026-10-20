"""按 pitch 标定夹爪开口区域（板端运行，画面远程同步到 PC 上操作）。

每个 pitch 角度标一次：空格冻结画面 -> 按顺序点 4 个点 -> 回车 -> 终端输入 pitch 原始值。
4 个点必须每次按同一顺序点，运行时才能逐点插值：
  1 左上  2 右上  3 右下  4 左下   （夹爪内侧开口，方块进夹爪后所在的区域）
建议在 -25° ~ +25° 范围内每隔约 10° 标一次（如 -2500/-1500/-500/500/1500/2500）。

按键：空格 冻结/恢复  u 撤销点  r 清空点  回车 保存当前角度  p 预览插值  d 删除最近一条  q 保存退出
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ORDER = ("左上", "右上", "右下", "左下")


def load_table(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"pitch_unit": "0.01deg", "entries": []}  # 不写 null，板端用 cv::FileStorage 读


def interpolate(entries, pitch):
    """与 gripper_guard.hpp 中 PitchRoiTable::roiAt 相同的逐点线性插值，超出范围取端点。"""
    pts = sorted(entries, key=lambda e: e["pitch"])
    if pitch <= pts[0]["pitch"]:
        return np.array(pts[0]["points"], np.float32)
    if pitch >= pts[-1]["pitch"]:
        return np.array(pts[-1]["points"], np.float32)
    for a, b in zip(pts, pts[1:]):
        if a["pitch"] <= pitch <= b["pitch"]:
            t = (pitch - a["pitch"]) / max(1, b["pitch"] - a["pitch"])
            return (1 - t) * np.array(a["points"], np.float32) + t * np.array(b["points"], np.float32)


def ask_pitch(prompt):
    while True:
        text = input(prompt).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            print("请输入整数原始值，例如 1500 表示 15°；直接回车取消")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="摄像头索引或设备路径")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", type=Path, default=Path("gripper_roi.json"))
    args = ap.parse_args()

    cam = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 {args.camera}")

    table = load_table(args.out)
    points, frozen, frame, preview = [], False, None, None
    win = "pitch ROI calib"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: points.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN and frozen and len(points) < 4 else None)
    print(__doc__)
    print(f"已有 {len(table['entries'])} 条：", sorted(e["pitch"] for e in table["entries"]))

    while True:
        if not frozen:
            ok, grabbed = cap.read()
            if not ok:
                print("读取画面失败")
                break
            frame = grabbed
        view = frame.copy()
        if preview is not None:
            cv2.polylines(view, [preview.astype(np.int32)], True, (0, 255, 255), 2)
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(points) >= 2:
            cv2.polylines(view, [np.array(points, np.int32)], len(points) == 4, (0, 0, 255), 2)
        hint = f"click {len(points) + 1}/4" if frozen and len(points) < 4 else ("Enter=save" if frozen else "SPACE=freeze")
        cv2.putText(view, f"{hint}  entries={len(table['entries'])}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            frozen, points = not frozen, []
            if frozen:
                print(f"已冻结，请依次点：{' -> '.join(ORDER)}")
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10):
            if len(points) != 4:
                print(f"需要 4 个点，当前 {len(points)} 个")
                continue
            pitch = ask_pitch("当前 pitch 原始值（如 1500，回车取消）：")
            if pitch is None:
                continue
            h, w = frame.shape[:2]
            if table.get("image_size") and table["image_size"] != [w, h]:
                print(f"分辨率 {w}x{h} 与已有标定 {table['image_size']} 不一致，未保存")
                continue
            table["image_size"] = [w, h]
            table["entries"] = [e for e in table["entries"] if e["pitch"] != pitch]  # 同角度重标则覆盖
            table["entries"].append({"pitch": pitch, "points": [list(map(int, p)) for p in points]})
            table["entries"].sort(key=lambda e: e["pitch"])
            args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已保存 pitch={pitch}，共 {len(table['entries'])} 条 -> {args.out}")
            frozen, points = False, []
        elif key == ord("p") and table["entries"]:
            pitch = ask_pitch("预览哪个 pitch 原始值（回车关闭预览）：")
            preview = None if pitch is None else interpolate(table["entries"], pitch)
        elif key == ord("d") and table["entries"]:
            removed = table["entries"].pop()
            args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已删除 pitch={removed['pitch']}")
        elif key in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
