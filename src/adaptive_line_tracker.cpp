// AdaptiveLineTracker implementation with pimpl pattern

#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
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

  // Check which contour a segment belongs to
  int find_segment_contour(const std::vector<std::vector<cv::Point>>& contours,
                           const Segment& seg, int y);

  // Check if two segments belong to the same contour
  bool are_segments_in_same_contour(
      const std::vector<std::vector<cv::Point>>& contours, const Segment& seg1,
      int y1, const Segment& seg2, int y2);

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

  // Step 1: Debug output (every frame)
  static int frame_count = 0;
  bool debug_this_frame = true;  // Debug every frame
  frame_count++;
  if (debug_this_frame) {
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

  // Step 2: Find contours in the mask for connectivity checking
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat mask_copy = black_mask.clone();  // findContours modifies the input
  cv::findContours(mask_copy, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  if (debug_this_frame) {
    std::cout << "Found " << contours.size() << " contours" << std::endl;
    // Show contour details
    for (size_t i = 0; i < contours.size() && i < 5; i++) {
      cv::Rect bbox = cv::boundingRect(contours[i]);
      std::cout << "  Contour " << i << ": " << contours[i].size() << " points"
                << ", bbox: [" << bbox.x << "," << bbox.y << "," << bbox.width
                << "x" << bbox.height << "]"
                << ", area: " << cv::contourArea(contours[i]) << std::endl;
    }
  }

  // Step 3: Initialize tracking structures - group by contour
  // Map from contour index to collected points
  std::map<int, std::vector<cv::Point2d>> contour_points;

  // Step 4: Scan from bottom to top and collect points by contour
  int total_scans = 0;
  int successful_detections = 0;
  const int scan_step = config_.scan_step;

  for (int y = black_mask.rows - 2; y > 0; y -= scan_step) {
    auto segments = find_black_segments(black_mask, y);
    total_scans++;

    if (segments.empty()) {
      continue;
    }

    successful_detections++;

    // Debug: Show segment to contour mapping
    if (debug_this_frame && (y > 450 || y % 50 == 3)) {
      std::cout << "[SEGMENT-CONTOUR] y=" << y << ": ";
      for (size_t i = 0; i < segments.size(); i++) {
        int contour_idx = find_segment_contour(contours, segments[i], y);
        std::cout << "seg" << i << "[" << segments[i].start_x << "-"
                  << segments[i].end_x << "]->C" << contour_idx << " ";
      }
      std::cout << std::endl;
    }

    // Group segments by contour
    for (const auto& seg : segments) {
      int contour_idx = find_segment_contour(contours, seg, y);
      if (contour_idx >= 0) {
        // Add point to the corresponding contour group
        contour_points[contour_idx].push_back(cv::Point2d(seg.center(), y));
      }
    }

    if (debug_this_frame && y > 450) {
      std::cout << "[DEBUG] y=" << y << ": " << segments.size() << " segments";
      // Show details of each segment
      for (size_t i = 0; i < segments.size() && i < 3;
           i++) {  // Show up to 3 segments
        std::cout << " | seg" << i << "[" << segments[i].start_x << "-"
                  << segments[i].end_x << "]w=" << segments[i].width();
      }
      std::cout << std::endl;
    }
  }

  // Step 5: Filter contour groups and collect valid points
  std::vector<cv::Point2d> candidate_points;
  const size_t MIN_POINTS_PER_CONTOUR =
      5;  // Minimum points to consider a contour valid

  if (debug_this_frame) {
    std::cout << "[TRACK DEBUG] Processing " << contour_points.size()
              << " contour groups:" << std::endl;
  }

  for (const auto& [contour_idx, points] : contour_points) {
    if (debug_this_frame) {
      std::cout << "  Contour " << contour_idx << ": " << points.size()
                << " points";
      if (!points.empty()) {
        std::cout << ", y-range: [" << points.back().y << " - "
                  << points.front().y << "]";
      }
    }

    if (points.size() >= MIN_POINTS_PER_CONTOUR) {
      // Add all points from valid contours
      for (const auto& pt : points) {
        candidate_points.push_back(pt);
      }
      if (debug_this_frame) {
        std::cout << " -> VALID (added " << points.size() << " points)";
      }
    } else {
      if (debug_this_frame) {
        std::cout << " -> SKIPPED (too few points)";
      }
    }

    if (debug_this_frame) {
      std::cout << std::endl;
    }
  }

  // Sort by y coordinate (descending, since bottom is higher y value)
  std::sort(candidate_points.begin(), candidate_points.end(),
            [](const cv::Point2d& a, const cv::Point2d& b) {
              return a.y > b.y;  // Bottom to top
            });

  if (debug_this_frame) {
    std::cout << "  Combined into " << candidate_points.size()
              << " total points" << std::endl;
  }

  // Step 6: Report tracking statistics
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

  // Step 7: Return results (without smoothing)
  // Note: Smoothing is disabled as it causes issues with branches
  // where points from different branches get averaged together
  tracked_points_ = candidate_points;

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
    bool is_black_line =
        mask.at<uchar>(y, x) >
        128;  // White pixels (255) in mask represent black line

    if (is_black_line && !in_black) {
      start_x = x;
      in_black = true;
    } else if (!is_black_line && in_black) {
      double width = x - start_x;
      if (width >= config_.min_line_width) {  // Only check minimum width
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
    if (width >= config_.min_line_width) {  // Only check minimum width
      // Store in relative coordinates
      segments.push_back({start_x, end_x});
    }
  }

  return segments;
}

int AdaptiveLineTracker::Impl::find_segment_contour(
    const std::vector<std::vector<cv::Point>>& contours, const Segment& seg,
    int y) {
  // Check center point of segment
  cv::Point test_point(seg.center(), y);

  // Debug: Show what we're testing (occasionally)
  static int call_count = 0;
  bool debug_this_call = (call_count++ % 50 == 0);

  if (debug_this_call) {
    std::cout << "[FIND_CONTOUR] Testing segment [" << seg.start_x << "-"
              << seg.end_x << "] at y=" << y << ", center=" << seg.center()
              << std::endl;
  }

  // Find which contour contains this point
  for (size_t i = 0; i < contours.size(); i++) {
    double result = cv::pointPolygonTest(contours[i], test_point, false);
    if (debug_this_call && i < 3) {  // Show first 3 contours
      std::cout << "  Contour " << i << ": pointPolygonTest = " << result
                << std::endl;
    }
    if (result >= 0) {
      if (debug_this_call) {
        std::cout << "  -> Matched contour " << i << " (center test)"
                  << std::endl;
      }
      return i;
    }
  }

  // Also check endpoints if center didn't match
  cv::Point start_point(seg.start_x, y);
  cv::Point end_point(seg.end_x, y);

  for (size_t i = 0; i < contours.size(); i++) {
    if (cv::pointPolygonTest(contours[i], start_point, false) >= 0 ||
        cv::pointPolygonTest(contours[i], end_point, false) >= 0) {
      if (debug_this_call) {
        std::cout << "  -> Matched contour " << i << " (endpoint test)"
                  << std::endl;
      }
      return i;
    }
  }

  if (debug_this_call) {
    std::cout << "  -> No contour found!" << std::endl;
  }

  return -1;  // No contour found
}

bool AdaptiveLineTracker::Impl::are_segments_in_same_contour(
    const std::vector<std::vector<cv::Point>>& contours, const Segment& seg1,
    int y1, const Segment& seg2, int y2) {
  int contour1 = find_segment_contour(contours, seg1, y1);
  int contour2 = find_segment_contour(contours, seg2, y2);

  // Debug output for troubleshooting
  static int debug_count = 0;
  if (debug_count++ % 20 == 0) {  // Every 20 calls
    std::cout << "[CONTOUR MATCH] seg1[" << seg1.start_x << "-" << seg1.end_x
              << "]@y=" << y1 << " -> contour " << contour1 << ", seg2["
              << seg2.start_x << "-" << seg2.end_x << "]@y=" << y2
              << " -> contour " << contour2 << " => "
              << (contour1 >= 0 && contour1 == contour2 ? "MATCH" : "NO MATCH")
              << std::endl;
  }

  // Both segments must belong to the same valid contour
  return (contour1 >= 0 && contour1 == contour2);
}
