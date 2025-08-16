// AdaptiveLineTracker implementation with pimpl pattern

#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

// Implementation class
class AdaptiveLineTracker::Impl {
 public:
  explicit Impl(rclcpp::Node* node);
  ~Impl() = default;

  std::vector<AdaptiveLineTracker::TrackedLine> track_line(
      const cv::Mat& black_mask, double total_processing_ms = -1.0);
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

  // Segment with scoring information
  struct SegmentInfo {
    cv::Point2d center;
    double width;
    double score;  // Score based on width and position
  };

  // Contour evaluation score
  struct ContourScore {
    int segment_count;
    double weighted_score;  // Total weighted score
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

  // Calculate score for a segment based on width and position
  double calculate_segment_score(const Segment& seg, int y, int image_height,
                                 int image_width);

  // Collect segments and calculate scores for each contour
  std::map<int, ContourScore> collect_and_score_contours(
      const cv::Mat& black_mask,
      const std::vector<std::vector<cv::Point>>& contours,
      std::map<int, std::vector<SegmentInfo>>& contour_segments);

  // Check if a contour is valid for tracking based on its score
  bool is_contour_valid_for_tracking(const ContourScore& score);

  // Build TrackedLine structures from collected segments
  std::vector<AdaptiveLineTracker::TrackedLine> build_tracked_lines(
      const std::map<int, std::vector<SegmentInfo>>& contour_segments,
      const std::map<int, ContourScore>& contour_scores,
      const std::vector<std::vector<cv::Point>>& contours);

  // Member variables
  std::vector<cv::Point2d> tracked_points_;
  double confidence_;
  Config config_;
  rclcpp::Node* node_;  // Node pointer for logging
};

// AdaptiveLineTracker
AdaptiveLineTracker::AdaptiveLineTracker(rclcpp::Node* node)
    : pimpl(std::make_unique<Impl>(node)) {}

AdaptiveLineTracker::~AdaptiveLineTracker() = default;

std::vector<AdaptiveLineTracker::TrackedLine> AdaptiveLineTracker::track_line(
    const cv::Mat& black_mask, double total_processing_ms) {
  return pimpl->track_line(black_mask, total_processing_ms);
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

AdaptiveLineTracker::Impl::Impl(rclcpp::Node* node)
    : confidence_(0.0), node_(node) {}

void AdaptiveLineTracker::Impl::reset() {
  tracked_points_.clear();
  confidence_ = 0.0;
}

std::vector<AdaptiveLineTracker::TrackedLine>
AdaptiveLineTracker::Impl::track_line(const cv::Mat& black_mask,
                                      double total_processing_ms) {
  tracked_points_.clear();
  std::vector<AdaptiveLineTracker::TrackedLine> result;

  if (black_mask.empty()) {
    return result;
  }

  // Timing start
  const auto t_start = std::chrono::steady_clock::now();

  // Step 1: Frame counter for debugging
  static int frame_count = 0;
  frame_count++;

  // Step 2: Find contours in the mask for connectivity checking
  std::vector<std::vector<cv::Point>> all_contours;
  cv::Mat mask_copy = black_mask.clone();  // findContours modifies the input
  cv::findContours(mask_copy, all_contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // Filter out small contours
  const double MIN_CONTOUR_AREA = 20.0;  // Skip contours smaller than this
  std::vector<std::vector<cv::Point>> contours;
  for (const auto& contour : all_contours) {
    double area = cv::contourArea(contour);
    if (area >= MIN_CONTOUR_AREA) {
      contours.push_back(contour);
    }
  }

  // Step 3: Collect segments and calculate scores for each contour
  std::map<int, std::vector<SegmentInfo>> contour_segments;
  auto contour_scores =
      collect_and_score_contours(black_mask, contours, contour_segments);

  // Step 4: Build TrackedLine structures using scoring
  result = build_tracked_lines(contour_segments, contour_scores, contours);

  // Step 5: Count statistics for logging
  int total_scans = 0;
  int successful_detections = 0;
  std::vector<int> all_segment_counts;

  // Count total scans and successful detections
  for (int y = black_mask.rows - 2; y > 0; y -= config_.scan_step) {
    total_scans++;
    auto segments = find_black_segments(black_mask, y);
    if (!segments.empty()) {
      successful_detections++;
    }
  }

  // Count segments per contour for logging
  for (size_t i = 0; i < contours.size(); i++) {
    auto it = contour_segments.find(i);
    if (it != contour_segments.end()) {
      all_segment_counts.push_back(it->second.size());
    } else {
      all_segment_counts.push_back(0);
    }
  }

  // Log summary every frame with timing
  if (node_) {
    // Calculate detection processing time
    const auto t_end = std::chrono::steady_clock::now();
    const double detection_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t_end - t_start)
            .count();

    // Build points string showing all contours with segments
    std::string points_info = " [points:";
    for (size_t i = 0; i < all_segment_counts.size(); i++) {
      if (i > 0) points_info += ",";
      points_info += std::to_string(all_segment_counts[i]);
    }
    points_info += "]";

    // Include total processing time if provided
    if (total_processing_ms >= 0) {
      RCLCPP_INFO(
          node_->get_logger(),
          "Detection: %d/%d scans, %zu/%zu contours valid%s (total: %.2f ms)",
          successful_detections, total_scans, contours.size(),
          all_contours.size(), points_info.c_str(), total_processing_ms);
    } else {
      RCLCPP_INFO(node_->get_logger(),
                  "Detection: %d/%d scans, %zu/%zu contours valid%s (%.2f ms)",
                  successful_detections, total_scans, contours.size(),
                  all_contours.size(), points_info.c_str(), detection_ms);
    }
  }

  // Step 7: Return results
  // Note: Keep tracked_points_ for backward compatibility if needed
  tracked_points_.clear();
  for (const auto& line : result) {
    for (const auto& pt : line.points) {
      tracked_points_.push_back(pt);
    }
  }

  return result;
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

  // Find which contour contains this point
  for (size_t i = 0; i < contours.size(); i++) {
    double result = cv::pointPolygonTest(contours[i], test_point, false);
    if (result >= 0) {
      return i;
    }
  }

  // Also check endpoints if center didn't match
  cv::Point start_point(seg.start_x, y);
  cv::Point end_point(seg.end_x, y);

  for (size_t i = 0; i < contours.size(); i++) {
    if (cv::pointPolygonTest(contours[i], start_point, false) >= 0 ||
        cv::pointPolygonTest(contours[i], end_point, false) >= 0) {
      return i;
    }
  }

  return -1;  // No contour found
}

bool AdaptiveLineTracker::Impl::are_segments_in_same_contour(
    const std::vector<std::vector<cv::Point>>& contours, const Segment& seg1,
    int y1, const Segment& seg2, int y2) {
  int contour1 = find_segment_contour(contours, seg1, y1);
  int contour2 = find_segment_contour(contours, seg2, y2);

  // Both segments must belong to the same valid contour
  return (contour1 >= 0 && contour1 == contour2);
}

double AdaptiveLineTracker::Impl::calculate_segment_score(const Segment& seg,
                                                          int y,
                                                          int image_height,
                                                          int image_width) {
  double width = seg.width();
  double relative_width = width / image_width;  // Relative width to image

  // Width-based weight
  // - Narrow segments (straight line): base score 1.0
  // - Wide segments (curve): higher score based on width
  double width_weight = 1.0 + (relative_width * config_.width_importance);

  // Position weight (bottom of image is more important)
  double position_weight = 1.0 + (1.0 - (double)y / image_height) * 0.5;

  return width_weight * position_weight;
}

std::map<int, AdaptiveLineTracker::Impl::ContourScore>
AdaptiveLineTracker::Impl::collect_and_score_contours(
    const cv::Mat& black_mask,
    const std::vector<std::vector<cv::Point>>& contours,
    std::map<int, std::vector<SegmentInfo>>& contour_segments) {
  std::map<int, ContourScore> contour_scores;
  const int scan_step = config_.scan_step;

  // Scan from bottom to top
  for (int y = black_mask.rows - 2; y > 0; y -= scan_step) {
    auto segments = find_black_segments(black_mask, y);

    for (const auto& seg : segments) {
      int contour_idx = find_segment_contour(contours, seg, y);
      if (contour_idx >= 0 && contour_idx < static_cast<int>(contours.size())) {
        // Calculate segment score
        double score =
            calculate_segment_score(seg, y, black_mask.rows, black_mask.cols);

        // Create SegmentInfo
        SegmentInfo seg_info;
        seg_info.center = cv::Point2d(seg.center(), y);
        seg_info.width = seg.width();
        seg_info.score = score;

        // Add to contour segments
        contour_segments[contour_idx].push_back(seg_info);

        // Update contour score
        contour_scores[contour_idx].segment_count++;
        contour_scores[contour_idx].weighted_score += score;
      }
    }
  }

  return contour_scores;
}

bool AdaptiveLineTracker::Impl::is_contour_valid_for_tracking(
    const ContourScore& score) {
  // Method 1: Check weighted score
  if (score.weighted_score >= config_.min_contour_score) {
    return true;
  }

  // Method 2: Adaptive threshold based on segment count
  // For curves with wide segments, accept fewer segments
  if (score.segment_count >= config_.min_segments_straight) {
    return true;
  }

  // Check if this might be a curve (high score with fewer segments)
  double average_score_per_segment =
      score.segment_count > 0 ? score.weighted_score / score.segment_count : 0;
  if (score.segment_count >= config_.min_segments_curve &&
      average_score_per_segment > 2.0) {
    return true;
  }

  return false;
}

std::vector<AdaptiveLineTracker::TrackedLine>
AdaptiveLineTracker::Impl::build_tracked_lines(
    const std::map<int, std::vector<SegmentInfo>>& contour_segments,
    const std::map<int, ContourScore>& contour_scores,
    const std::vector<std::vector<cv::Point>>& contours) {
  std::vector<AdaptiveLineTracker::TrackedLine> result;

  for (size_t i = 0; i < contours.size(); i++) {
    auto score_it = contour_scores.find(i);

    // Create TrackedLine for this contour
    AdaptiveLineTracker::TrackedLine tracked_line;
    tracked_line.contour_id = i;
    tracked_line.area = cv::contourArea(contours[i]);

    // Check if this contour has segments and is valid
    auto seg_it = contour_segments.find(i);
    if (seg_it != contour_segments.end() && score_it != contour_scores.end()) {
      // Check if valid for tracking
      if (is_contour_valid_for_tracking(score_it->second)) {
        // Extract points from SegmentInfo
        for (const auto& seg_info : seg_it->second) {
          tracked_line.points.push_back(seg_info.center);
        }

        // Sort points by y coordinate (descending)
        std::sort(tracked_line.points.begin(), tracked_line.points.end(),
                  [](const cv::Point2d& a, const cv::Point2d& b) {
                    return a.y > b.y;  // Bottom to top
                  });

        result.push_back(tracked_line);
      }
    }
  }

  return result;
}
