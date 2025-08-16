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

  // Check if two segments are vertically connected
  bool are_segments_connected(const cv::Mat& mask, const Segment& seg1, int y1,
                              const Segment& seg2, int y2);

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

  // Step 1: Debug output (every 30 frames)
  static int frame_count = 0;
  bool debug_this_frame = (frame_count++ % 30 == 0);
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

  // Step 2: Initialize tracking structures
  struct TrackedLine {
    std::vector<cv::Point2d> points;  // Detected points for this line
    Segment last_segment;             // Last detected segment
    int last_y;                       // Y coordinate of last detection
    int consecutive_missing;          // Count of consecutive missing detections
  };

  std::vector<TrackedLine> tracked_lines;
  std::vector<cv::Point2d> candidate_points;

  // Step 3: Scan from bottom to top
  int total_scans = 0;
  int successful_detections = 0;
  const int scan_step = config_.scan_step;

  for (int y = black_mask.rows - 2; y > 0; y -= scan_step) {
    auto segments = find_black_segments(black_mask, y);
    total_scans++;

    if (segments.empty()) {
      // Update consecutive missing count for all tracked lines
      for (auto& line : tracked_lines) {
        line.consecutive_missing++;
      }
      continue;
    }

    successful_detections++;

    // Match segments to existing tracked lines or create new ones
    std::vector<bool> segment_matched(segments.size(), false);

    // Try to match each tracked line with a segment
    for (auto& line : tracked_lines) {
      if (line.consecutive_missing >
          10) {    // Increased tolerance for missing segments
        continue;  // Skip lines that have been missing too long
      }

      int best_match_idx = -1;
      double best_overlap = 0;

      // Find best matching segment based on vertical connectivity
      for (size_t i = 0; i < segments.size(); i++) {
        if (segment_matched[i]) continue;

        // Check if segments are connected through black pixels
        if (are_segments_connected(black_mask, line.last_segment, line.last_y,
                                   segments[i], y)) {
          // Calculate overlap ratio
          double overlap_start =
              std::max(line.last_segment.start_x, segments[i].start_x);
          double overlap_end =
              std::min(line.last_segment.end_x, segments[i].end_x);
          double overlap = std::max(0.0, overlap_end - overlap_start);
          double overlap_ratio = overlap / std::min(line.last_segment.width(),
                                                    segments[i].width());

          if (overlap_ratio > best_overlap) {
            best_overlap = overlap_ratio;
            best_match_idx = i;
          }
        }
      }

      if (best_match_idx >= 0) {
        // Found a match - continue this line
        segment_matched[best_match_idx] = true;
        line.points.push_back(
            cv::Point2d(segments[best_match_idx].center(), y));
        line.last_segment = segments[best_match_idx];
        line.last_y = y;
        line.consecutive_missing = 0;
      } else {
        // No match found
        line.consecutive_missing++;
      }
    }

    // Create new tracked lines for unmatched segments
    for (size_t i = 0; i < segments.size(); i++) {
      if (!segment_matched[i]) {
        TrackedLine new_line;
        new_line.points.push_back(cv::Point2d(segments[i].center(), y));
        new_line.last_segment = segments[i];
        new_line.last_y = y;
        new_line.consecutive_missing = 0;
        tracked_lines.push_back(new_line);
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
      std::cout << " | " << tracked_lines.size() << " tracked lines"
                << std::endl;
    }
  }

  // Step 4: Filter and select significant lines
  if (!tracked_lines.empty()) {
    // Filter out short lines that are likely noise
    const size_t MIN_POINTS_FOR_LINE = 8;  // ~40 pixels at scan_step=5
    std::vector<size_t> significant_lines;

    for (size_t i = 0; i < tracked_lines.size(); i++) {
      if (tracked_lines[i].points.size() >= MIN_POINTS_FOR_LINE) {
        significant_lines.push_back(i);
      }
    }

    if (!significant_lines.empty()) {
      // Combine all significant lines into candidate_points
      // Sort them by y-coordinate to maintain bottom-up order
      for (size_t idx : significant_lines) {
        for (const auto& pt : tracked_lines[idx].points) {
          candidate_points.push_back(pt);
        }
      }

      // Sort by y coordinate (descending, since bottom is higher y value)
      std::sort(candidate_points.begin(), candidate_points.end(),
                [](const cv::Point2d& a, const cv::Point2d& b) {
                  return a.y > b.y;  // Bottom to top
                });

      if (debug_this_frame) {
        std::cout << "[TRACK DEBUG] Found " << significant_lines.size()
                  << " significant lines (>= 5 points)" << std::endl;
        for (size_t idx : significant_lines) {
          std::cout << "  Line " << idx << ": "
                    << tracked_lines[idx].points.size() << " points";
          if (!tracked_lines[idx].points.empty()) {
            std::cout << ", y-range: [" << tracked_lines[idx].points.back().y
                      << " - " << tracked_lines[idx].points.front().y << "]";
          }
          std::cout << std::endl;
        }
        std::cout << "  Combined into " << candidate_points.size()
                  << " total points" << std::endl;
      }
    }
  }

  // Step 5: Report tracking statistics
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

  // Step 6: Return results (without smoothing)
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

bool AdaptiveLineTracker::Impl::are_segments_connected(const cv::Mat& mask,
                                                       const Segment& seg1,
                                                       int y1,
                                                       const Segment& seg2,
                                                       int y2) {
  // Check connectivity by analyzing the black region shape around seg1
  // and checking if seg2 falls within the expected continuation

  if (y1 == y2) {
    return false;  // Same scan line - should not happen
  }

  // Ensure y1 is below y2 (y1 > y2 since we scan bottom-up)
  if (y1 < y2) {
    return are_segments_connected(mask, seg2, y2, seg1, y1);
  }

  // Analyze the black region shape around seg1 to predict where it continues
  // Start from seg1 and trace the black region upward

  // Define the region of interest around seg1
  int margin = 5;  // Pixels to check on each side
  int x_start = std::max(0, seg1.start_x - margin);
  int x_end = std::min(mask.cols - 1, seg1.end_x + margin);

  // Trace the black region from y1 towards y2
  std::vector<std::pair<int, int>>
      traced_boundaries;  // (left_x, right_x) for each y

  for (int y = y1; y >= y2 && y >= 0; y--) {
    int left_boundary = -1;
    int right_boundary = -1;

    // Find the leftmost and rightmost black pixels in this row
    // Start search from previous boundaries if available
    if (!traced_boundaries.empty()) {
      auto& prev = traced_boundaries.back();
      // Search window based on previous boundaries with some tolerance
      int search_start = std::max(0, prev.first - 10);
      int search_end = std::min(mask.cols - 1, prev.second + 10);

      // Find left boundary
      for (int x = search_start; x <= search_end; x++) {
        if (mask.at<uchar>(y, x) > 128) {  // Black pixel
          left_boundary = x;
          break;
        }
      }

      // Find right boundary
      if (left_boundary >= 0) {
        for (int x = search_end; x >= left_boundary; x--) {
          if (mask.at<uchar>(y, x) > 128) {  // Black line pixel (white in mask)
            right_boundary = x;
            break;
          }
        }
      }
    } else {
      // First iteration - search in the original segment range
      for (int x = x_start; x <= x_end; x++) {
        if (mask.at<uchar>(y, x) > 128) {
          if (left_boundary < 0) left_boundary = x;
          right_boundary = x;
        }
      }
    }

    // Check if we lost the line
    if (left_boundary < 0 || right_boundary < 0) {
      // No black pixels found - line discontinued
      break;
    }

    traced_boundaries.push_back({left_boundary, right_boundary});

    // If we reached y2, check if seg2 overlaps with the traced region
    if (y == y2) {
      // Check if seg2 overlaps with the traced boundary at y2
      int overlap_start = std::max(seg2.start_x, left_boundary);
      int overlap_end = std::min(seg2.end_x, right_boundary);

      if (overlap_start <= overlap_end) {
        // Calculate overlap ratio
        double overlap = overlap_end - overlap_start + 1;
        double seg2_width = seg2.width();
        double traced_width = right_boundary - left_boundary + 1;
        double min_width = std::min(seg2_width, traced_width);

        // Consider connected if significant overlap (>50% of smaller segment)
        return (overlap / min_width) > 0.5;
      }

      // Check if seg2 is very close to the traced boundary
      int gap = std::min(std::abs(seg2.end_x - left_boundary),
                         std::abs(right_boundary - seg2.start_x));
      return gap <= 5;  // Allow small gaps
    }
  }

  // If we couldn't trace all the way to y2, segments are not connected
  return false;
}
