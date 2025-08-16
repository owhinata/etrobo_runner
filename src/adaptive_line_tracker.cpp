// AdaptiveLineTracker implementation with pimpl pattern

#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

// =======================
// Implementation class
// =======================
class AdaptiveLineTracker::Impl {
 public:
  Impl();
  ~Impl() = default;

  std::vector<cv::Point2d> track_line(const cv::Mat& black_mask);
  void reset();
  double get_confidence() const { return confidence_; }
  void set_config(const Config& config) { config_ = config; }
  Config get_config() const { return config_; }

 private:
  // Segment structure for black regions
  struct Segment {
    int start_x;
    int end_x;

    double center() const { return (start_x + end_x) / 2.0; }
    double width() const { return end_x - start_x; }
  };

  // Find black segments in a horizontal scan line
  std::vector<Segment> find_black_segments(const cv::Mat& mask, int y);

  // Smooth the tracked trajectory
  std::vector<cv::Point2d> smooth_trajectory(
      const std::vector<cv::Point2d>& points);

  // Member variables
  std::vector<cv::Point2d> tracked_points_;
  double confidence_;
  Config config_;
};

// =======================
// AdaptiveLineTracker
// =======================
AdaptiveLineTracker::AdaptiveLineTracker() : pimpl(std::make_unique<Impl>()) {}

AdaptiveLineTracker::~AdaptiveLineTracker() = default;

std::vector<cv::Point2d> AdaptiveLineTracker::track_line(
    const cv::Mat& black_mask) {
  return pimpl->track_line(black_mask);
}

void AdaptiveLineTracker::reset() { pimpl->reset(); }

double AdaptiveLineTracker::get_confidence() const {
  return pimpl->get_confidence();
}

void AdaptiveLineTracker::set_config(const Config& config) {
  pimpl->set_config(config);
}

AdaptiveLineTracker::Config AdaptiveLineTracker::get_config() const {
  return pimpl->get_config();
}

// =======================
// Impl implementation
// =======================
AdaptiveLineTracker::Impl::Impl() : confidence_(0.0) {}

void AdaptiveLineTracker::Impl::reset() {
  tracked_points_.clear();
  confidence_ = 0.0;
}

std::vector<cv::Point2d> AdaptiveLineTracker::Impl::track_line(
    const cv::Mat& black_mask) {
  tracked_points_.clear();

  if (black_mask.empty()) {
    return tracked_points_;
  }

  // Debug: Check overall black pixel distribution
  static int frame_count = 0;
  bool debug_this_frame = (frame_count++ % 30 == 0);
  if (debug_this_frame) {  // Every 30 frames
    std::cout << "\n[TRACK DEBUG] Frame " << frame_count << std::endl;
    std::cout << "Image size: " << black_mask.cols << "x" << black_mask.rows
              << std::endl;

    // Check black pixels at different y levels
    for (int y = black_mask.rows - 10; y > 0; y -= 50) {
      int black_count = 0;
      for (int x = 0; x < black_mask.cols; x++) {
        if (black_mask.at<uchar>(y, x) > 128) black_count++;
      }
      std::cout << "  y=" << y << ": " << black_count << " black pixels"
                << std::endl;
    }
  }

  std::vector<cv::Point2d> candidate_points;

  // Simple approach: NO TRACKING - just detect black segments at each scan line
  // Bottom-up scanning from robot position upward
  int total_scans = 0;
  int successful_detections = 0;

  // Use fixed scan step for simplicity
  const int scan_step = config_.scan_step;

  for (int y = black_mask.rows - 2; y > 0; y -= scan_step) {
    auto segments = find_black_segments(black_mask, y);
    total_scans++;

    if (segments.empty()) {
      continue;  // Skip to next scan line
    }

    successful_detections++;

    // Simple selection: just take the largest segment (most likely to be the
    // main line)
    int best_idx = 0;
    double max_width = segments[0].width();
    for (size_t i = 1; i < segments.size(); i++) {
      if (segments[i].width() > max_width) {
        max_width = segments[i].width();
        best_idx = i;
      }
    }

    if (debug_this_frame && y > 450) {
      std::cout << "[DEBUG] y=" << y << ": " << segments.size() << " segments";
      std::cout << ", chosen center=" << segments[best_idx].center()
                << ", width=" << segments[best_idx].width() << std::endl;
    }

    // Add the center point of the best segment
    double center_x = segments[best_idx].center();
    cv::Point2d current_point(center_x, y);
    candidate_points.push_back(current_point);
  }

  // Debug: Report tracking statistics
  if (debug_this_frame) {
    std::cout << "[TRACK DEBUG] Scanning complete:" << std::endl;
    std::cout << "  Total scans: " << total_scans << std::endl;
    std::cout << "  Successful detections: " << successful_detections
              << std::endl;
    std::cout << "  Candidate points: " << candidate_points.size() << std::endl;

    if (!candidate_points.empty()) {
      std::cout << "  First point: (" << candidate_points.front().x << ", "
                << candidate_points.front().y << ")" << std::endl;
      std::cout << "  Last point: (" << candidate_points.back().x << ", "
                << candidate_points.back().y << ")" << std::endl;
    }
  }

  // Apply minimal smoothing to reduce noise
  if (candidate_points.size() > static_cast<size_t>(config_.smooth_window)) {
    tracked_points_ = smooth_trajectory(candidate_points);
  } else {
    tracked_points_ = candidate_points;
  }

  return tracked_points_;
}

std::vector<AdaptiveLineTracker::Impl::Segment>
AdaptiveLineTracker::Impl::find_black_segments(const cv::Mat& mask, int y) {
  std::vector<Segment> segments;

  // Check if y is within the mask bounds
  if (y < 0 || y >= mask.rows) {
    return segments;
  }

  bool in_black = false;
  int start_x = 0;

  // Scan horizontally at row y
  for (int x = 0; x < mask.cols; x++) {
    bool is_black = mask.at<uchar>(y, x) > 128;

    if (is_black && !in_black) {
      start_x = x;
      in_black = true;
    } else if (!is_black && in_black) {
      double width = x - start_x;
      if (width >= config_.min_line_width && width <= config_.max_line_width) {
        // Store in relative coordinates
        segments.push_back({start_x, x});
      }
      in_black = false;
    }
  }

  // Handle segment that extends to edge of mask
  if (in_black) {
    int end_x = mask.cols;
    double width = end_x - start_x;
    if (width >= config_.min_line_width && width <= config_.max_line_width) {
      // Store in relative coordinates
      segments.push_back({start_x, end_x});
    }
  }

  return segments;
}

std::vector<cv::Point2d> AdaptiveLineTracker::Impl::smooth_trajectory(
    const std::vector<cv::Point2d>& points) {
  if (points.size() < 3) {  // Need at least 3 points for smoothing
    return points;
  }

  std::vector<cv::Point2d> smoothed;
  smoothed.reserve(points.size());

  // Simple 3-point moving average smoothing
  for (size_t i = 0; i < points.size(); i++) {
    if (i == 0 || i == points.size() - 1) {
      // Keep edge points as-is
      smoothed.push_back(points[i]);
    } else {
      // Average with neighbors
      double avg_x = (points[i - 1].x + points[i].x + points[i + 1].x) / 3.0;
      double avg_y = (points[i - 1].y + points[i].y + points[i + 1].y) / 3.0;
      smoothed.push_back(cv::Point2d(avg_x, avg_y));
    }
  }

  return smoothed;
}
