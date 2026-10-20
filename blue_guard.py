"""HSV 蓝色兜底：模型把蓝色物块认成 orange/green/black 时，否决对该框的夹取。

摄像头随 pitch 移动，夹爪在画面中的位置不固定，所以不用固定 ROI，
只检查"准备去夹的那个框"本身：框中心区域的蓝色像素占比够高，就认为它其实是蓝块。

在 blocks_v1 标注上的实测（center_ratio=0.4）：
  真蓝块命中 290/300（96.7%）；非蓝目标被误否决 44/2395（1.8%），
  其中 38 个在 011309/011605 两段蓝色放置区视频里（黑色三角块框中心透出蓝色地面）。

只依赖 cv2 + numpy，可直接拷到板端与控制代码一起用。
"""
import cv2
import numpy as np

# 蓝色物块 H≈92–99；V 下限放到 40，暗光下的蓝块（V≈52）也能命中
BLUE_LOW = (90, 120, 40)
BLUE_HIGH = (115, 255, 255)


def blue_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def center_blue_ratio(mask, box):
    """框中心 50%×50% 区域内的蓝色像素占比。box 为原图像素 (x1, y1, x2, y2)。"""
    h, w = mask.shape
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    qw, qh = max(1.0, x2 - x1) / 4, max(1.0, y2 - y1) / 4
    core = mask[max(0, int(cy - qh)):min(h, int(cy + qh) + 1),
                max(0, int(cx - qw)):min(w, int(cx + qw) + 1)]
    return float((core > 0).mean()) if core.size else 0.0


def is_blue_target(mask, box, center_ratio=0.4):
    """True 表示这个框很可能是蓝块，不能夹。"""
    return center_blue_ratio(mask, box) >= center_ratio
