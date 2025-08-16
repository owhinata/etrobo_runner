// AdaptiveLineTracker implementation with pimpl pattern

#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

#include "src/line_detector_node.hpp"

// Implementation class
class AdaptiveLineTracker::Impl {
 public:
  explicit Impl(LineDetectorNode* node);
  ~Impl() = default;

  void declare_parameters();
  bool try_update_parameter(const rclcpp::Parameter& param);
  bool process_frame(const cv::Mat& img,
                     AdaptiveLineTracker::DetectionResult& result);
  void draw_visualization_overlay(cv::Mat& img) const;
  void reset();
  double get_confidence() const { return confidence_; }
  void set_config(const Config& config) { config_ = config; }
  Config get_config() const { return config_; }

 private:
  // Helper function for ROI validation
  cv::Rect valid_roi(const cv::Mat& img) const {
    if (roi_.size() != 4) return cv::Rect(0, 0, img.cols, img.rows);
    int x = static_cast<int>(roi_[0]);
    int y = static_cast<int>(roi_[1]);
    int w = static_cast<int>(roi_[2]);
    int h = static_cast<int>(roi_[3]);
    if (x < 0 || y < 0 || w <= 0 || h <= 0) {
      return cv::Rect(0, 0, img.cols, img.rows);
    }
    x = std::max(0, std::min(x, img.cols - 1));
    y = std::max(0, std::min(y, img.rows - 1));
    w = std::min(w, img.cols - x);
    h = std::min(h, img.rows - y);
    return cv::Rect(x, y, w, h);
  }

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

  // Extract black regions from image using HSV thresholding
  cv::Mat extract_black_regions(const cv::Mat& img);

  // Member variables
  LineDetectorNode* node_;  // Node pointer for parameters and logging
  std::vector<cv::Point2d> tracked_points_;
  double confidence_{0.0};
  Config config_;  // Already has default values in struct definition

  // HSV parameters for black line detection
  int hsv_upper_v_{100};
  int hsv_dilate_kernel_{3};
  int hsv_dilate_iter_{1};
  bool show_contours_{false};

  // ROI for processing
  std::vector<int64_t> roi_;

  // Last detection result for visualization
  AdaptiveLineTracker::DetectionResult last_result_;
};

// AdaptiveLineTracker
AdaptiveLineTracker::AdaptiveLineTracker(LineDetectorNode* node)
    : pimpl(std::make_unique<Impl>(node)) {}

AdaptiveLineTracker::~AdaptiveLineTracker() = default;

void AdaptiveLineTracker::declare_parameters() { pimpl->declare_parameters(); }

bool AdaptiveLineTracker::try_update_parameter(const rclcpp::Parameter& param) {
  return pimpl->try_update_parameter(param);
}

bool AdaptiveLineTracker::process_frame(const cv::Mat& img,
                                        DetectionResult& result) {
  return pimpl->process_frame(img, result);
}

void AdaptiveLineTracker::draw_visualization_overlay(cv::Mat& img) const {
  pimpl->draw_visualization_overlay(img);
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

AdaptiveLineTracker::Impl::Impl(LineDetectorNode* node) : node_(node) {}

void AdaptiveLineTracker::Impl::reset() {
  tracked_points_.clear();
  confidence_ = 0.0;
  last_result_ = AdaptiveLineTracker::DetectionResult();
}

void AdaptiveLineTracker::Impl::declare_parameters() {
  // Image preprocessing
  roi_ = node_->declare_parameter<std::vector<int64_t>>("roi",
                                                        std::vector<int64_t>{});

  // HSV mask parameters for black line detection
  hsv_upper_v_ = node_->declare_parameter<int>("hsv_upper_v", 100);
  hsv_dilate_kernel_ = node_->declare_parameter<int>("hsv_dilate_kernel", 3);
  hsv_dilate_iter_ = node_->declare_parameter<int>("hsv_dilate_iter", 1);

  // Visualization parameters
  show_contours_ = node_->declare_parameter<bool>("show_contours", false);

  // Line tracking configuration parameters
  config_.scan_step = node_->declare_parameter<int>("line_scan_step", 5);
  config_.min_line_width =
      node_->declare_parameter<double>("min_line_width", 6.0);
  config_.max_line_width =
      node_->declare_parameter<double>("max_line_width", 50.0);
  config_.width_importance =
      node_->declare_parameter<double>("line_width_importance", 2.0);
  config_.min_contour_score =
      node_->declare_parameter<double>("min_contour_score", 10.0);
  config_.min_segments_straight =
      node_->declare_parameter<int>("min_segments_straight", 5);
  config_.min_segments_curve =
      node_->declare_parameter<int>("min_segments_curve", 3);
}

bool AdaptiveLineTracker::Impl::try_update_parameter(
    const rclcpp::Parameter& param) {
  const std::string& name = param.get_name();

  if (name == "roi") {
    roi_ = param.as_integer_array();
    return true;
  } else if (name == "hsv_upper_v") {
    hsv_upper_v_ = param.as_int();
    return true;
  } else if (name == "hsv_dilate_kernel") {
    hsv_dilate_kernel_ = param.as_int();
    return true;
  } else if (name == "hsv_dilate_iter") {
    hsv_dilate_iter_ = param.as_int();
    return true;
  } else if (name == "show_contours") {
    show_contours_ = param.as_bool();
    return true;
  } else if (name == "line_scan_step") {
    config_.scan_step = param.as_int();
    return true;
  } else if (name == "min_line_width") {
    config_.min_line_width = param.as_double();
    return true;
  } else if (name == "max_line_width") {
    config_.max_line_width = param.as_double();
    return true;
  } else if (name == "line_width_importance") {
    config_.width_importance = param.as_double();
    return true;
  } else if (name == "min_contour_score") {
    config_.min_contour_score = param.as_double();
    return true;
  } else if (name == "min_segments_straight") {
    config_.min_segments_straight = param.as_int();
    return true;
  } else if (name == "min_segments_curve") {
    config_.min_segments_curve = param.as_int();
    return true;
  }

  return false;
}

bool AdaptiveLineTracker::Impl::process_frame(
    const cv::Mat& img, AdaptiveLineTracker::DetectionResult& result) {
  tracked_points_.clear();

  // Clear result
  result.tracked_lines.clear();
  result.segment_counts.clear();
  result.total_scans = 0;
  result.successful_detections = 0;
  result.total_contours = 0;
  result.valid_contours = 0;

  // Clear last result
  last_result_ = AdaptiveLineTracker::DetectionResult();

  if (img.empty()) {
    return false;
  }

  // Step 1: Extract ROI and convert to black mask
  cv::Rect roi_rect = valid_roi(img);
  cv::Mat work_img = img(roi_rect).clone();
  cv::Mat black_mask = extract_black_regions(work_img);

  if (black_mask.empty()) {
    return false;
  }

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
  result.tracked_lines =
      build_tracked_lines(contour_segments, contour_scores, contours);

  // Step 5: Populate statistics for result
  result.total_contours = all_contours.size();
  result.valid_contours = contours.size();
  result.total_scans = 0;
  result.successful_detections = 0;

  // Count total scans and successful detections
  for (int y = black_mask.rows - 2; y > 0; y -= config_.scan_step) {
    result.total_scans++;
    auto segments = find_black_segments(black_mask, y);
    if (!segments.empty()) {
      result.successful_detections++;
    }
  }

  // Count segments per contour
  for (size_t i = 0; i < contours.size(); i++) {
    auto it = contour_segments.find(i);
    if (it != contour_segments.end()) {
      result.segment_counts.push_back(it->second.size());
    } else {
      result.segment_counts.push_back(0);
    }
  }

  // Step 6: Convert points to absolute coordinates (add ROI offset)
  for (auto& line : result.tracked_lines) {
    for (auto& pt : line.points) {
      pt.x += roi_rect.x;
      pt.y += roi_rect.y;
    }
  }

  // Step 7: Update internal tracked points for backward compatibility
  tracked_points_.clear();
  for (const auto& line : result.tracked_lines) {
    for (const auto& pt : line.points) {
      tracked_points_.push_back(pt);
    }
  }

  // Store result for visualization
  last_result_ = result;

  return !result.tracked_lines.empty();
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

cv::Mat AdaptiveLineTracker::Impl::extract_black_regions(const cv::Mat& img) {
  cv::Mat black_mask;

  // Convert to HSV if needed
  cv::Mat hsv;
  if (img.channels() == 3) {
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  } else {
    // For grayscale, create pseudo-HSV with V channel only
    cv::Mat channels[3];
    channels[0] = cv::Mat::zeros(img.size(), CV_8UC1);  // H = 0
    channels[1] = cv::Mat::zeros(img.size(), CV_8UC1);  // S = 0
    channels[2] = img;                                  // V = grayscale value
    cv::merge(channels, 3, hsv);
  }

  // Extract black regions (low V value)
  cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, hsv_upper_v_),
              black_mask);

  // Apply morphological operations to clean up
  if (hsv_dilate_iter_ > 0 && hsv_dilate_kernel_ > 0) {
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(hsv_dilate_kernel_, hsv_dilate_kernel_));

    // Remove small noise first
    cv::morphologyEx(black_mask, black_mask, cv::MORPH_OPEN, kernel);

    // Then close small gaps in the line
    cv::morphologyEx(black_mask, black_mask, cv::MORPH_CLOSE, kernel);

    // Additional erosion to remove edge noise
    cv::erode(black_mask, black_mask, kernel, cv::Point(-1, -1), 1);
  }

  return black_mask;
}

void AdaptiveLineTracker::Impl::draw_visualization_overlay(cv::Mat& img) const {
  // Draw ROI rectangle
  cv::Rect roi_rect = valid_roi(img);
  cv::rectangle(img, roi_rect, cv::Scalar(255, 255, 0), 1);

  // Draw contours if enabled
  if (show_contours_) {
    // Get contours from the current black mask
    cv::Mat work_img = img(roi_rect).clone();
    cv::Mat black_mask =
        const_cast<Impl*>(this)->extract_black_regions(work_img);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(black_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);

    // Draw only valid contours (area >= 20)
    const double MIN_CONTOUR_AREA = 20.0;
    int valid_contour_idx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
      double area = cv::contourArea(contours[i]);
      if (area < MIN_CONTOUR_AREA) {
        continue;
      }

      // Create offset contour for correct positioning
      std::vector<std::vector<cv::Point>> contour_to_draw;
      std::vector<cv::Point> offset_contour;
      for (const auto& pt : contours[i]) {
        offset_contour.push_back(
            cv::Point(pt.x + roi_rect.x, pt.y + roi_rect.y));
      }
      contour_to_draw.push_back(offset_contour);

      // Use different colors for different valid contours
      cv::Scalar contour_color;
      switch (valid_contour_idx % 6) {
        case 0:
          contour_color = cv::Scalar(255, 0, 0);
          break;  // Blue
        case 1:
          contour_color = cv::Scalar(0, 255, 255);
          break;  // Yellow
        case 2:
          contour_color = cv::Scalar(255, 0, 255);
          break;  // Magenta
        case 3:
          contour_color = cv::Scalar(0, 255, 128);
          break;  // Green-yellow
        case 4:
          contour_color = cv::Scalar(255, 128, 0);
          break;  // Orange
        case 5:
          contour_color = cv::Scalar(128, 0, 255);
          break;  // Purple
      }

      cv::drawContours(img, contour_to_draw, 0, contour_color, 2);
      valid_contour_idx++;
    }
  }

  // Draw tracked points with different colors for different contours
  for (const auto& line : last_result_.tracked_lines) {
    // Use different colors for different contours
    cv::Scalar point_color;
    switch (line.contour_id % 6) {
      case 0:
        point_color = cv::Scalar(0, 255, 0);
        break;  // Green
      case 1:
        point_color = cv::Scalar(0, 165, 255);
        break;  // Orange
      case 2:
        point_color = cv::Scalar(255, 255, 0);
        break;  // Cyan
      case 3:
        point_color = cv::Scalar(255, 0, 255);
        break;  // Magenta
      case 4:
        point_color = cv::Scalar(0, 255, 255);
        break;  // Yellow
      case 5:
        point_color = cv::Scalar(255, 128, 128);
        break;  // Light Blue
    }

    for (const auto& pt : line.points) {
      // Points are already in absolute coordinates
      cv::Point center(static_cast<int>(pt.x), static_cast<int>(pt.y));
      cv::circle(img, center, 3, point_color, -1);
    }
  }
}
