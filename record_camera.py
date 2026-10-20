import argparse
import time

from datetime import datetime
from pathlib import Path

import cv2


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 {index}，请检查 /dev/video{index} 是否存在或被其他程序占用")
    # MJPG 才能在 USB2.0 下跑满 720p 30fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def new_writer(out_dir: Path, prefix, size, fps):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"无法创建视频文件：{path}")
    return writer, path


def main():
    ap = argparse.ArgumentParser(description="USB 摄像头录像：空格开始/停止录制，q 或 Esc 退出")
    ap.add_argument("--camera", type=int, default=0, help="摄像头序号，0 对应 /dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--output-dir", type=Path, default=Path("videos"))
    ap.add_argument("--prefix", default="run", help="视频文件名前缀")
    ap.add_argument("--duration", type=float, default=0, help="每段最长秒数，到时自动停止；0 表示不限")
    ap.add_argument("--no-preview", action="store_true", help="不显示窗口，启动即录制，按 Ctrl+C 或到时长结束")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"camera={args.camera} size={width}x{height} fps={fps:g}")

    writer, path, started = None, None, 0.0
    frames = 0

    def start():
        nonlocal writer, path, started, frames
        writer, path = new_writer(args.output_dir, args.prefix, (width, height), fps)
        started, frames = time.time(), 0
        print(f"开始录制：{path}")

    def stop():
        nonlocal writer
        if writer is not None:
            writer.release()
            print(f"已保存：{path}（{frames} 帧，{time.time() - started:.1f} 秒）")
            writer = None

    if args.no_preview:
        start()
    else:
        print("空格：开始/停止录制    q / Esc：退出")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取画面失败，摄像头可能已断开")
                break
            if writer is not None:
                writer.write(frame)
                frames += 1
                if args.duration and time.time() - started >= args.duration:
                    stop()
                    if args.no_preview:
                        break
            if args.no_preview:
                continue

            view = frame.copy()
            if writer is not None:
                cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(view, f"REC {time.time() - started:5.1f}s", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(view, "SPACE: record   Q: quit", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("record_camera", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                stop() if writer is not None else start()
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
