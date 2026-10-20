// 夹取前的安全检查（板端 C++，仅依赖 OpenCV）。
//
// 1. 蓝色兜底：目标框中心 50%x50% 区域蓝色像素占比 >= centerRatio 就否决（与 blue_guard.py 相同）。
// 2. 夹爪区域：按下位机回传的 pitch（0.01°，1500 = 15°）在 gripper_roi.json 的标定点之间逐点线性插值，
//    得到当前画面里的夹爪开口四边形；只接受面积一半以上落在区域内、且短边够大的框。
//    最小短边按区域宽度的比例给，pitch 变化导致画面缩放时阈值跟着变。
// 3. 连续帧确认：同一判断连续 N 帧成立才输出，避免单帧闪烁触发动作。
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace gripper_guard {

// ---------- 1. HSV 蓝色兜底 ----------
inline cv::Mat blueMask(const cv::Mat& bgr) {
    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    // 蓝色物块 H≈92–99；V 下限 40，暗光下的蓝块（V≈52）也能命中
    cv::inRange(hsv, cv::Scalar(90, 120, 40), cv::Scalar(115, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    return mask;
}

inline double centerBlueRatio(const cv::Mat& mask, const cv::Rect2f& box) {
    const float cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    const float qw = std::max(1.f, box.width) / 4, qh = std::max(1.f, box.height) / 4;
    cv::Rect core(cv::Point(int(cx - qw), int(cy - qh)), cv::Point(int(cx + qw) + 1, int(cy + qh) + 1));
    core &= cv::Rect(0, 0, mask.cols, mask.rows);
    if (core.area() <= 0) return 0.0;
    return double(cv::countNonZero(mask(core))) / core.area();
}

inline bool isBlueTarget(const cv::Mat& mask, const cv::Rect2f& box, double centerRatio = 0.4) {
    return centerBlueRatio(mask, box) >= centerRatio;
}

// ---------- 2. 按 pitch 插值的夹爪区域 ----------
class PitchRoiTable {
public:
    // 读取 pitch_roi_calib.py 生成的 gripper_roi.json
    void load(const std::string& path) {
        cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) throw std::runtime_error("cannot open " + path);
        entries_.clear();
        for (const auto& node : fs["entries"]) {
            Entry e;
            e.pitch = int(node["pitch"]);
            for (const auto& p : node["points"]) e.points.emplace_back(float(p[0]), float(p[1]));
            if (e.points.size() != 4) throw std::runtime_error("each entry needs 4 points");
            entries_.push_back(e);
        }
        if (entries_.empty()) throw std::runtime_error("no entries in " + path);
        std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.pitch < b.pitch; });
    }

    // pitch 为下位机原始值（0.01°）；超出标定范围取端点
    std::vector<cv::Point2f> roiAt(int pitch) const {
        if (pitch <= entries_.front().pitch) return entries_.front().points;
        if (pitch >= entries_.back().pitch) return entries_.back().points;
        for (size_t i = 1; i < entries_.size(); ++i) {
            const Entry &a = entries_[i - 1], &b = entries_[i];
            if (pitch > b.pitch) continue;
            const float t = float(pitch - a.pitch) / std::max(1, b.pitch - a.pitch);
            std::vector<cv::Point2f> out(4);
            for (int k = 0; k < 4; ++k) out[k] = a.points[k] * (1 - t) + b.points[k] * t;
            return out;
        }
        return entries_.back().points;
    }

    bool empty() const { return entries_.empty(); }

private:
    struct Entry {
        int pitch = 0;
        std::vector<cv::Point2f> points;
    };
    std::vector<Entry> entries_;
};

// 框与四边形的重叠面积 / 框面积
inline double overlapRatio(const cv::Rect2f& box, const std::vector<cv::Point2f>& roi) {
    std::vector<cv::Point2f> rect = {box.tl(), {box.br().x, box.y}, box.br(), {box.x, box.br().y}};
    std::vector<cv::Point2f> poly = roi, inter;
    if (cv::contourArea(poly, true) < 0) std::reverse(poly.begin(), poly.end());  // intersectConvexConvex 需同向
    const float area = cv::intersectConvexConvex(rect, poly, inter, true);
    return box.area() > 0 ? area / box.area() : 0.0;
}

struct Params {
    double centerRatio = 0.4;     // 蓝色兜底阈值
    double minOverlap = 0.5;      // 框至少一半面积在夹爪区域内
    double minSideOfRoi = 0.15;   // 框短边 >= 区域宽度 * 该比例（实测：真方块 ≥0.28，误报 0.06–0.09）
    int confirmFrames = 3;        // 连续多少帧成立才确认
};

enum class Verdict { None, Grab, BlueVeto };

struct Detection {
    cv::Rect2f box;
    int classId;
    float score;
};

// ---------- 3. 每帧调用：综合判断 + 连续帧确认 ----------
class GripperGuard {
public:
    GripperGuard(const PitchRoiTable& table, int blueClassId, Params params = {})
        : table_(table), blueClassId_(blueClassId), p_(params) {}

    // dets：模型输出（原图坐标）；pitch：下位机最新回传值。
    // 返回连续 confirmFrames 帧一致后的结论；BlueVeto 优先于 Grab。
    Verdict update(const cv::Mat& bgr, const std::vector<Detection>& dets, int pitch, int* targetIndex = nullptr) {
        const std::vector<cv::Point2f> roi = table_.roiAt(pitch);
        const float roiWidth = std::max(cv::norm(roi[1] - roi[0]), cv::norm(roi[2] - roi[3]));
        const cv::Mat mask = blueMask(bgr);

        Verdict now = Verdict::None;
        int best = -1;
        for (size_t i = 0; i < dets.size(); ++i) {
            const Detection& d = dets[i];
            if (overlapRatio(d.box, roi) < p_.minOverlap) continue;
            if (std::min(d.box.width, d.box.height) < p_.minSideOfRoi * roiWidth) continue;
            // 夹爪区域内出现蓝块（模型认出的，或被认成别的颜色但 HSV 判为蓝色）一律否决
            if (d.classId == blueClassId_ || isBlueTarget(mask, d.box, p_.centerRatio)) {
                now = Verdict::BlueVeto;
                best = int(i);
                break;
            }
            if (best < 0 || d.score > dets[best].score) {
                now = Verdict::Grab;
                best = int(i);
            }
        }

        streak_ = (now == last_) ? streak_ + 1 : 1;
        last_ = now;
        if (targetIndex) *targetIndex = best;
        if (now == Verdict::BlueVeto) return Verdict::BlueVeto;  // 否决不等待确认，立即生效
        return streak_ >= p_.confirmFrames ? now : Verdict::None;
    }

private:
    const PitchRoiTable& table_;
    int blueClassId_;
    Params p_;
    Verdict last_ = Verdict::None;
    int streak_ = 0;
};

}  // namespace gripper_guard
