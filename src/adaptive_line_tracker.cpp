// AdaptiveLineTracker implementation with pimpl pattern

#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

#include "src/branch_merge_handler.hpp"
#include "src/contour_tracker.hpp"
#include "src/line_detector_node.hpp"
#include "src/simple_line_selector.hpp"

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

  // Extract black and colored regions from image using HSV thresholding
  cv::Mat extract_line_regions(const cv::Mat& img);

  // Process contours: find, track, and filter
  std::vector<std::vector<cv::Point>> process_contours(const cv::Mat& mask);

  // Build final detection result from scored contours
  void build_detection_result(
      const std::vector<std::vector<cv::Point>>& contours,
      const std::map<int, std::vector<SegmentInfo>>& contour_segments,
      const std::map<int, ContourScore>& contour_scores,
      const cv::Mat& black_mask, const cv::Rect& roi_rect,
      AdaptiveLineTracker::DetectionResult& result);

  // Member variables
  LineDetectorNode* node_;  // Node pointer for parameters and logging
  double confidence_{0.0};
  Config config_;  // Already has default values in struct definition

  // HSV parameters for black line detection
  int hsv_lower_s_{0};
  int hsv_upper_s_{255};
  int hsv_upper_v_{100};
  int hsv_dilate_kernel_{3};
  int hsv_dilate_iter_{1};
  bool show_contours_{false};
  bool show_mask_{false};

  // HSV parameters for blue line detection
  bool blue_detection_enabled_{true};
  int blue_lower_h_{100};  // Blue hue range: 100-130
  int blue_upper_h_{130};
  int blue_lower_s_{50};  // Minimum saturation for blue
  int blue_upper_s_{255};
  int blue_lower_v_{50};  // Minimum value for blue
  int blue_upper_v_{255};

  // HSV parameters for gray disk detection
  bool gray_detection_enabled_{true};
  int gray_upper_s_{16};   // Low saturation for gray (0-16)
  int gray_lower_v_{100};  // Minimum brightness for gray disk
  int gray_upper_v_{168};  // Maximum brightness for gray disk

  // ROI for processing
  std::vector<int64_t> roi_;

  // Last detection result for visualization
  AdaptiveLineTracker::DetectionResult last_result_;

  // Last black mask for visualization
  cv::Mat last_black_mask_;

  // Contour tracker for temporal tracking
  std::unique_ptr<ContourTracker> contour_tracker_;

  // Branch/merge handler for handling line topology
  std::unique_ptr<BranchMergeHandler> branch_merge_handler_;
  std::map<int, std::vector<BranchMergeHandler::Segment>>
      previous_contour_segments_;

  // Simple line selector as alternative
  std::unique_ptr<SimpleLineSelector> simple_selector_;
  bool use_simple_selector_{false};  // Flag to enable simple selector
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

AdaptiveLineTracker::Impl::Impl(LineDetectorNode* node) : node_(node) {
  contour_tracker_ = std::make_unique<ContourTracker>();
  // Configure tracker
  contour_tracker_->set_max_missed_frames(5);
  contour_tracker_->set_max_distance_threshold(75.0);  // Increased for curves
  contour_tracker_->set_min_contour_area(20.0);

  // Initialize branch/merge handler
  branch_merge_handler_ = std::make_unique<BranchMergeHandler>();

  // Initialize simple selector
  simple_selector_ = std::make_unique<SimpleLineSelector>(10);
}

void AdaptiveLineTracker::Impl::reset() {
  confidence_ = 0.0;
  last_result_ = AdaptiveLineTracker::DetectionResult();
  previous_contour_segments_.clear();
  if (simple_selector_) {
    simple_selector_->reset();
  }
}

void AdaptiveLineTracker::Impl::declare_parameters() {
  // Image preprocessing
  roi_ = node_->declare_parameter<std::vector<int64_t>>("roi",
                                                        std::vector<int64_t>{});

  // HSV mask parameters for black line detection
  hsv_lower_s_ = node_->declare_parameter<int>("hsv_lower_s", 0);
  hsv_upper_s_ = node_->declare_parameter<int>("hsv_upper_s", 255);
  hsv_upper_v_ = node_->declare_parameter<int>("hsv_upper_v", 100);
  hsv_dilate_kernel_ = node_->declare_parameter<int>("hsv_dilate_kernel", 3);
  hsv_dilate_iter_ = node_->declare_parameter<int>("hsv_dilate_iter", 1);

  // Visualization parameters
  show_contours_ = node_->declare_parameter<bool>("show_contours", false);
  show_mask_ = node_->declare_parameter<bool>("show_mask", false);

  // Blue line detection parameters
  blue_detection_enabled_ =
      node_->declare_parameter<bool>("blue_detection_enabled", true);
  blue_lower_h_ = node_->declare_parameter<int>("blue_lower_h", 100);
  blue_upper_h_ = node_->declare_parameter<int>("blue_upper_h", 130);
  blue_lower_s_ = node_->declare_parameter<int>("blue_lower_s", 50);
  blue_upper_s_ = node_->declare_parameter<int>("blue_upper_s", 255);
  blue_lower_v_ = node_->declare_parameter<int>("blue_lower_v", 50);
  blue_upper_v_ = node_->declare_parameter<int>("blue_upper_v", 255);

  // Gray disk detection parameters (based on CameraCalibrator settings)
  gray_detection_enabled_ =
      node_->declare_parameter<bool>("gray_detection_enabled", true);
  gray_upper_s_ = node_->declare_parameter<int>("gray_upper_s",
                                                16);  // Low saturation for gray
  gray_lower_v_ =
      node_->declare_parameter<int>("gray_lower_v", 100);  // Min brightness
  gray_upper_v_ =
      node_->declare_parameter<int>("gray_upper_v", 168);  // Max brightness

  // Tracking parameters
  bool tracker_enabled =
      node_->declare_parameter<bool>("tracker_enabled", true);
  int tracker_max_missed =
      node_->declare_parameter<int>("tracker_max_missed_frames", 5);
  double tracker_max_dist = node_->declare_parameter<double>(
      "tracker_max_distance", 75.0);  // Increased for curves
  bool tracker_debug = node_->declare_parameter<bool>("tracker_debug", false);
  double tracker_process_noise =
      node_->declare_parameter<double>("tracker_process_noise", 1e-2);
  double tracker_measurement_noise =
      node_->declare_parameter<double>("tracker_measurement_noise", 5e-2);
  double tracker_speed_threshold =
      node_->declare_parameter<double>("tracker_speed_threshold", 5.0);

  // Configure tracker with parameters
  if (contour_tracker_) {
    contour_tracker_->set_enabled(tracker_enabled);
    contour_tracker_->set_max_missed_frames(tracker_max_missed);
    contour_tracker_->set_max_distance_threshold(tracker_max_dist);
    contour_tracker_->set_debug_enabled(tracker_debug);
    contour_tracker_->set_process_noise(tracker_process_noise);
    contour_tracker_->set_measurement_noise(tracker_measurement_noise);
    contour_tracker_->set_speed_threshold(tracker_speed_threshold);
  }

  // Line tracking configuration parameters
  config_.scan_step = node_->declare_parameter<int>("line_scan_step", 5);
  config_.min_line_width =
      node_->declare_parameter<double>("min_line_width", 6.0);
  config_.width_importance =
      node_->declare_parameter<double>("line_width_importance", 2.0);
  config_.min_contour_score =
      node_->declare_parameter<double>("min_contour_score", 10.0);
  config_.min_segments_straight =
      node_->declare_parameter<int>("min_segments_straight", 5);
  config_.min_segments_curve =
      node_->declare_parameter<int>("min_segments_curve", 3);

  // Branch/Merge handler parameters
  bool bmh_enabled =
      node_->declare_parameter<bool>("branch_merge_enabled", true);
  use_simple_selector_ = node_->declare_parameter<bool>(
      "use_simple_selector", true);  // Use simple selector by default
  std::string bmh_branch_strategy =
      node_->declare_parameter<std::string>("branch_strategy", "alternating");
  std::string bmh_merge_strategy =
      node_->declare_parameter<std::string>("merge_strategy", "continuity");
  double bmh_continuity_threshold =
      node_->declare_parameter<double>("continuity_threshold", 30.0);

  // Configure branch/merge handler
  if (branch_merge_handler_) {
    BranchMergeHandler::Config bmh_config;
    bmh_config.enabled = bmh_enabled;

    // Parse branch strategy
    if (bmh_branch_strategy == "left_priority") {
      bmh_config.branch_strategy = BranchMergeHandler::Config::LEFT_PRIORITY;
    } else if (bmh_branch_strategy == "right_priority") {
      bmh_config.branch_strategy = BranchMergeHandler::Config::RIGHT_PRIORITY;
    } else if (bmh_branch_strategy == "straight_priority") {
      bmh_config.branch_strategy =
          BranchMergeHandler::Config::STRAIGHT_PRIORITY;
    } else {
      bmh_config.branch_strategy = BranchMergeHandler::Config::ALTERNATING;
    }

    // Parse merge strategy
    if (bmh_merge_strategy == "width_based") {
      bmh_config.merge_strategy = BranchMergeHandler::Config::WIDTH_BASED;
    } else if (bmh_merge_strategy == "center_based") {
      bmh_config.merge_strategy = BranchMergeHandler::Config::CENTER_BASED;
    } else {
      bmh_config.merge_strategy = BranchMergeHandler::Config::CONTINUITY;
    }

    bmh_config.continuity_threshold = bmh_continuity_threshold;
    branch_merge_handler_->set_config(bmh_config);
  }
}

bool AdaptiveLineTracker::Impl::try_update_parameter(
    const rclcpp::Parameter& param) {
  const std::string& name = param.get_name();

  if (name == "roi") {
    roi_ = param.as_integer_array();
    return true;
  } else if (name == "hsv_lower_s") {
    hsv_lower_s_ = param.as_int();
    return true;
  } else if (name == "hsv_upper_s") {
    hsv_upper_s_ = param.as_int();
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
  } else if (name == "show_mask") {
    show_mask_ = param.as_bool();
    return true;
  } else if (name == "line_scan_step") {
    config_.scan_step = param.as_int();
    return true;
  } else if (name == "min_line_width") {
    config_.min_line_width = param.as_double();
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
  } else if (name == "blue_detection_enabled") {
    blue_detection_enabled_ = param.as_bool();
    return true;
  } else if (name == "blue_lower_h") {
    blue_lower_h_ = param.as_int();
    return true;
  } else if (name == "blue_upper_h") {
    blue_upper_h_ = param.as_int();
    return true;
  } else if (name == "blue_lower_s") {
    blue_lower_s_ = param.as_int();
    return true;
  } else if (name == "blue_upper_s") {
    blue_upper_s_ = param.as_int();
    return true;
  } else if (name == "blue_lower_v") {
    blue_lower_v_ = param.as_int();
    return true;
  } else if (name == "blue_upper_v") {
    blue_upper_v_ = param.as_int();
    return true;
  } else if (name == "gray_detection_enabled") {
    gray_detection_enabled_ = param.as_bool();
    return true;
  } else if (name == "gray_upper_s") {
    gray_upper_s_ = param.as_int();
    return true;
  } else if (name == "gray_lower_v") {
    gray_lower_v_ = param.as_int();
    return true;
  } else if (name == "gray_upper_v") {
    gray_upper_v_ = param.as_int();
    return true;
  }

  // Classifier selection
  if (name == "use_simple_selector") {
    use_simple_selector_ = param.as_bool();
    return true;
  }

  // Branch/Merge handler parameter updates
  if (branch_merge_handler_) {
    auto bmh_config = branch_merge_handler_->get_config();
    bool updated = false;

    if (name == "branch_merge_enabled") {
      bmh_config.enabled = param.as_bool();
      updated = true;
    } else if (name == "branch_strategy") {
      std::string strategy = param.as_string();
      if (strategy == "left_priority") {
        bmh_config.branch_strategy = BranchMergeHandler::Config::LEFT_PRIORITY;
      } else if (strategy == "right_priority") {
        bmh_config.branch_strategy = BranchMergeHandler::Config::RIGHT_PRIORITY;
      } else if (strategy == "straight_priority") {
        bmh_config.branch_strategy =
            BranchMergeHandler::Config::STRAIGHT_PRIORITY;
      } else {
        bmh_config.branch_strategy = BranchMergeHandler::Config::ALTERNATING;
      }
      updated = true;
    } else if (name == "merge_strategy") {
      std::string strategy = param.as_string();
      if (strategy == "width_based") {
        bmh_config.merge_strategy = BranchMergeHandler::Config::WIDTH_BASED;
      } else if (strategy == "center_based") {
        bmh_config.merge_strategy = BranchMergeHandler::Config::CENTER_BASED;
      } else {
        bmh_config.merge_strategy = BranchMergeHandler::Config::CONTINUITY;
      }
      updated = true;
    } else if (name == "continuity_threshold") {
      bmh_config.continuity_threshold = param.as_double();
      updated = true;
    }

    if (updated) {
      branch_merge_handler_->set_config(bmh_config);
      return true;
    }
  }

  // Tracker parameter updates
  if (!contour_tracker_) {
    return false;
  }

  if (name == "tracker_enabled") {
    contour_tracker_->set_enabled(param.as_bool());
    return true;
  } else if (name == "tracker_max_missed_frames") {
    contour_tracker_->set_max_missed_frames(param.as_int());
    return true;
  } else if (name == "tracker_max_distance") {
    contour_tracker_->set_max_distance_threshold(param.as_double());
    return true;
  } else if (name == "tracker_debug") {
    contour_tracker_->set_debug_enabled(param.as_bool());
    return true;
  } else if (name == "tracker_process_noise") {
    contour_tracker_->set_process_noise(param.as_double());
    return true;
  } else if (name == "tracker_measurement_noise") {
    contour_tracker_->set_measurement_noise(param.as_double());
    return true;
  } else if (name == "tracker_speed_threshold") {
    contour_tracker_->set_speed_threshold(param.as_double());
    return true;
  }

  return false;
}

bool AdaptiveLineTracker::Impl::process_frame(
    const cv::Mat& img, AdaptiveLineTracker::DetectionResult& result) {
  // Initialize
  result = AdaptiveLineTracker::DetectionResult();
  last_result_ = AdaptiveLineTracker::DetectionResult();

  if (img.empty()) {
    return false;
  }

  // Step 1: Extract ROI and line regions
  cv::Rect roi_rect = valid_roi(img);
  cv::Mat work_img = img(roi_rect).clone();
  last_black_mask_ = extract_line_regions(work_img);

  if (last_black_mask_.empty()) {
    return false;
  }

  // Step 2: Process contours (find, track, and filter)
  std::vector<std::vector<cv::Point>> contours =
      process_contours(last_black_mask_);

  // Step 3: Collect segments and calculate scores for each contour
  std::map<int, std::vector<SegmentInfo>> contour_segments;
  auto contour_scores =
      collect_and_score_contours(last_black_mask_, contours, contour_segments);

  // Step 4: Build final detection result
  build_detection_result(contours, contour_segments, contour_scores,
                         last_black_mask_, roi_rect, result);

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

    // Group segments by contour
    std::map<int, std::vector<BranchMergeHandler::Segment>> contour_to_segments;
    for (const auto& seg : segments) {
      int contour_idx = find_segment_contour(contours, seg, y);
      if (contour_idx >= 0 && contour_idx < static_cast<int>(contours.size())) {
        // Convert to BranchMergeHandler::Segment
        BranchMergeHandler::Segment bmh_seg{seg.start_x, seg.end_x};
        contour_to_segments[contour_idx].push_back(bmh_seg);
      }
    }

    // Process each contour's segments
    for (const auto& [contour_idx, segs] : contour_to_segments) {
      std::optional<BranchMergeHandler::Segment> selected_segment;

      // Use simple selector for primary contour
      if (use_simple_selector_ && segs.size() > 0 && contour_idx == 0) {
        // Convert to SimpleLineSelector segments
        std::vector<SimpleLineSelector::Segment> simple_segs;
        for (const auto& seg : segs) {
          SimpleLineSelector::Segment ss;
          ss.start_x = seg.start_x;
          ss.end_x = seg.end_x;
          ss.y = y;
          simple_segs.push_back(ss);
        }

        // Add scan to selector
        simple_selector_->add_scan(y, simple_segs);

        // Get selected segment
        auto selected_simple = simple_selector_->get_best_segment();
        if (selected_simple) {
          // Convert back to BranchMergeHandler::Segment
          BranchMergeHandler::Segment bmh_seg;
          bmh_seg.start_x = selected_simple->start_x;
          bmh_seg.end_x = selected_simple->end_x;
          selected_segment = bmh_seg;
        }
      } else {
        // Use original branch/merge handler
        BranchMergeHandler::SegmentContext context;
        context.current_segments = segs;
        context.previous_segments = previous_contour_segments_[contour_idx];
        context.contour_id = contour_idx;
        context.scan_y = y;
        context.scan_step = scan_step;

        // Let handler select the appropriate segment
        selected_segment = branch_merge_handler_->process_segments(context);
      }

      if (selected_segment) {
        // Calculate segment score
        Segment seg{selected_segment->start_x, selected_segment->end_x};
        double score =
            calculate_segment_score(seg, y, black_mask.rows, black_mask.cols);

        // Create SegmentInfo
        SegmentInfo seg_info;
        seg_info.center = cv::Point2d(selected_segment->center(), y);
        seg_info.width = selected_segment->width();
        seg_info.score = score;

        // Add to contour segments
        contour_segments[contour_idx].push_back(seg_info);

        // Update contour score
        contour_scores[contour_idx].segment_count++;
        contour_scores[contour_idx].weighted_score += score;
      }

      // Update previous segments for next iteration
      previous_contour_segments_[contour_idx] = segs;
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

std::vector<std::vector<cv::Point>> AdaptiveLineTracker::Impl::process_contours(
    const cv::Mat& mask) {
  // Find contours in the mask
  std::vector<std::vector<cv::Point>> all_contours;
  cv::Mat mask_copy = mask.clone();  // findContours modifies the input
  cv::findContours(mask_copy, all_contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // Update contour tracker with new contours
  contour_tracker_->update(all_contours);

  // Get all tracked contours
  const auto& all_tracked = contour_tracker_->get_tracked_contours();

  // Log tracking information if debug enabled
  if (!all_tracked.empty()) {
    std::string tracking_info = "Tracking: ";
    for (const auto& [id, tracked] : all_tracked) {
      tracking_info += "ID" + std::to_string(id) +
                       "(age:" + std::to_string(tracked.age) +
                       ",miss:" + std::to_string(tracked.missed_frames) + ") ";
    }
    RCLCPP_DEBUG(node_->get_logger(), "%s", tracking_info.c_str());
  }

  // Filter tracked contours by area and missed frames
  std::vector<std::vector<cv::Point>> filtered_contours;
  constexpr double MIN_CONTOUR_AREA = 20.0;
  filtered_contours.reserve(all_tracked.size());

  for (const auto& [id, tracked] : all_tracked) {
    if (tracked.area >= MIN_CONTOUR_AREA && tracked.missed_frames == 0) {
      filtered_contours.push_back(tracked.contour);
    }
  }

  return filtered_contours;
}

void AdaptiveLineTracker::Impl::build_detection_result(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::map<int, std::vector<SegmentInfo>>& contour_segments,
    const std::map<int, ContourScore>& contour_scores,
    const cv::Mat& black_mask, const cv::Rect& roi_rect,
    AdaptiveLineTracker::DetectionResult& result) {
  // Find the highest scoring contour
  result.best_contour_id = -1;
  result.best_contour_score = 0.0;
  result.best_contour.clear();

  for (const auto& [contour_id, score] : contour_scores) {
    if (score.weighted_score > result.best_contour_score) {
      result.best_contour_score = score.weighted_score;
      result.best_contour_id = contour_id;
      if (contour_id >= 0 && contour_id < static_cast<int>(contours.size())) {
        result.best_contour = contours[contour_id];
      }
    }
  }

  // Build TrackedLine structures using scoring
  result.tracked_lines =
      build_tracked_lines(contour_segments, contour_scores, contours);

  // Populate statistics for result
  // Note: We need to access all_contours which was local to process_contours
  // For now, use contours size as an approximation
  result.total_contours = contours.size();
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
  result.segment_counts.clear();
  result.segment_counts.reserve(contours.size());
  for (size_t i = 0; i < contours.size(); i++) {
    auto it = contour_segments.find(i);
    if (it != contour_segments.end()) {
      result.segment_counts.push_back(it->second.size());
    } else {
      result.segment_counts.push_back(0);
    }
  }

  // Convert points to absolute coordinates (add ROI offset)
  for (auto& line : result.tracked_lines) {
    for (auto& pt : line.points) {
      pt.x += roi_rect.x;
      pt.y += roi_rect.y;
    }
  }
}

cv::Mat AdaptiveLineTracker::Impl::extract_line_regions(const cv::Mat& img) {
  cv::Mat combined_mask;

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

  // Extract black regions (low V value, within S range)
  cv::Mat black_mask;
  cv::inRange(hsv, cv::Scalar(0, hsv_lower_s_, 0),
              cv::Scalar(180, hsv_upper_s_, hsv_upper_v_), black_mask);

  // Start with black mask
  combined_mask = black_mask.clone();

  // Extract blue regions if enabled
  if (blue_detection_enabled_) {
    cv::Mat blue_mask;
    cv::inRange(hsv, cv::Scalar(blue_lower_h_, blue_lower_s_, blue_lower_v_),
                cv::Scalar(blue_upper_h_, blue_upper_s_, blue_upper_v_),
                blue_mask);

    // Combine with existing mask
    cv::bitwise_or(combined_mask, blue_mask, combined_mask);
  }

  // Extract gray regions (gray disk) if enabled
  if (gray_detection_enabled_) {
    cv::Mat gray_mask;
    // Gray: Any hue, low saturation, specific brightness range
    cv::inRange(hsv, cv::Scalar(0, 0, gray_lower_v_),
                cv::Scalar(180, gray_upper_s_, gray_upper_v_), gray_mask);

    // Combine with existing mask
    cv::bitwise_or(combined_mask, gray_mask, combined_mask);
  }

  // Apply morphological operations to clean up
  if (hsv_dilate_iter_ > 0 && hsv_dilate_kernel_ > 0) {
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(hsv_dilate_kernel_, hsv_dilate_kernel_));

    // Remove small noise first
    cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_OPEN, kernel);

    // Then close small gaps in the line
    cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_CLOSE, kernel);

    // Additional erosion to remove edge noise
    cv::erode(combined_mask, combined_mask, kernel, cv::Point(-1, -1), 1);
  }

  return combined_mask;
}

void AdaptiveLineTracker::Impl::draw_visualization_overlay(cv::Mat& img) const {
  // Draw ROI rectangle
  cv::Rect roi_rect = valid_roi(img);
  cv::rectangle(img, roi_rect, cv::Scalar(255, 255, 0), 1);

  // Overlay black mask semi-transparently if show_mask is enabled
  if (show_mask_ && !last_black_mask_.empty()) {
    // Convert grayscale mask to BGR
    cv::Mat mask_bgr;
    cv::cvtColor(last_black_mask_, mask_bgr, cv::COLOR_GRAY2BGR);

    // Blend mask with original image (semi-transparent overlay)
    cv::Mat roi_img = img(roi_rect);
    cv::addWeighted(roi_img, 0.5, mask_bgr, 0.5, 0, roi_img);
  }

  // Draw contours if enabled (on top of mask if present)
  if (show_contours_ && contour_tracker_) {
    // Get all tracked contours
    const auto& tracked = contour_tracker_->get_tracked_contours();

    for (const auto& [id, tracked_contour] : tracked) {
      // Create offset contour for correct positioning
      std::vector<std::vector<cv::Point>> contour_to_draw;
      std::vector<cv::Point> offset_contour;
      for (const auto& pt : tracked_contour.contour) {
        offset_contour.push_back(
            cv::Point(pt.x + roi_rect.x, pt.y + roi_rect.y));
      }
      contour_to_draw.push_back(offset_contour);

      // Use consistent color based on ID
      cv::Scalar contour_color;
      switch (id % 6) {
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

      cv::drawContours(img, contour_to_draw, 0, contour_color, 1);

      // Draw ID and tracking info
      cv::Point2f centroid_offset(tracked_contour.centroid.x + roi_rect.x,
                                  tracked_contour.centroid.y + roi_rect.y);

      // Draw centroid
      cv::circle(img, centroid_offset, 4, contour_color, -1);

      // Draw ID text
      std::string text = "ID:" + std::to_string(id);
      if (tracked_contour.age > 1) {
        text += " (" + std::to_string(tracked_contour.age) + ")";
      }
      cv::putText(img, text,
                  cv::Point(centroid_offset.x + 5, centroid_offset.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, contour_color, 1);

      // Draw predicted position if missed
      if (tracked_contour.missed_frames > 0) {
        cv::Point2f pred_offset(
            tracked_contour.predicted_centroid.x + roi_rect.x,
            tracked_contour.predicted_centroid.y + roi_rect.y);
        cv::circle(img, pred_offset, 3, contour_color, 1);
        cv::line(img, centroid_offset, pred_offset, contour_color, 1);
      }
    }
  }

  // Draw the highest scoring contour with a special highlight
  if (last_result_.best_contour_id >= 0 && !last_result_.best_contour.empty()) {
    // Create offset contour for correct positioning
    std::vector<std::vector<cv::Point>> best_contour_to_draw;
    std::vector<cv::Point> offset_best_contour;
    for (const auto& pt : last_result_.best_contour) {
      offset_best_contour.push_back(
          cv::Point(pt.x + roi_rect.x, pt.y + roi_rect.y));
    }
    best_contour_to_draw.push_back(offset_best_contour);

    // Draw with thin white outline first
    cv::drawContours(img, best_contour_to_draw, 0, cv::Scalar(255, 255, 255),
                     2);
    // Then draw with bright green color
    cv::drawContours(img, best_contour_to_draw, 0, cv::Scalar(0, 255, 0), 1);
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
