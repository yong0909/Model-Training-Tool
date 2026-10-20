"""清洗 live_hardneg.py 收集的难负样本。

模型给的蓝框可能偏小，也可能漏检。用 HSV 蓝色掩膜做两件事：
1. 把每个蓝框扩展到与之相交的蓝色连通域，补全被截断的框；
2. 框外仍有大块蓝色（漏检）的帧整帧移到 rejected/，避免把蓝块当背景学。
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from live_hardneg import write_voc


def blue_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 120, 60), (115, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def refine(img, boxes, min_area):
    """返回 (修正后的框, 框外剩余蓝色像素数)。"""
    mask = blue_mask(img)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    out = []
    for name, (x1, y1, x2, y2) in boxes:
        bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
        region = labels[max(0, by1):by2, max(0, bx1):bx2]
        for comp in np.unique(region):
            if comp == 0 or stats[comp, cv2.CC_STAT_AREA] < min_area:
                continue
            cx, cy, cw, ch = stats[comp, :4]
            # 连通域只有不到 30% 落在框内时不并入，防止并到相邻的别的蓝色物体
            if (region == comp).sum() < 0.3 * stats[comp, cv2.CC_STAT_AREA]:
                continue
            x1, y1 = min(x1, cx), min(y1, cy)
            x2, y2 = max(x2, cx + cw), max(y2, cy + ch)
        out.append((name, (max(0, x1), max(0, y1), min(w, x2), min(h, y2))))
    stray = mask.copy()
    for _, (x1, y1, x2, y2) in out:
        stray[max(0, int(y1) - 15):int(y2) + 15, max(0, int(x1) - 15):int(x2) + 15] = 0
    return out, int((stray > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, help="blocks_v1_hardneg/<session> 目录")
    ap.add_argument("--stray-px", type=int, default=2500, help="框外蓝色像素超过该值视为漏检，整帧剔除")
    ap.add_argument("--min-area", type=int, default=400)
    args = ap.parse_args()

    rej = args.session / "rejected"
    (rej / "images").mkdir(parents=True, exist_ok=True)
    (rej / "annotations").mkdir(parents=True, exist_ok=True)
    kept = rejected = grown = 0
    for xml in sorted((args.session / "annotations").glob("*.xml")):
        jpg = args.session / "images" / f"{xml.stem}.jpg"
        img = cv2.imread(str(jpg))
        boxes = [(o.find("name").text, tuple(float(o.find("bndbox").find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")))
                 for o in ET.parse(xml).iter("object")]
        new, stray = refine(img, boxes, args.min_area)
        if stray > args.stray_px:
            shutil.move(str(jpg), rej / "images" / jpg.name)
            shutil.move(str(xml), rej / "annotations" / xml.name)
            rejected += 1
            print(f"reject {xml.stem} stray_blue={stray}")
            continue
        if any(abs(a - b) > 2 for (_, o), (_, n) in zip(boxes, new) for a, b in zip(o, n)):
            grown += 1
        h, w = img.shape[:2]
        write_voc(xml, jpg.name, w, h, new)
        kept += 1
    print(f"kept={kept} (boxes grown in {grown}) rejected={rejected}")


if __name__ == "__main__":
    main()
