// AdaptiveLineTracker implementation with pimpl pattern

#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

#include "src/contour_tracker.hpp"
#include "src/line_detector_node.hpp"
#include "src/line_merger.hpp"

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

  // Extract black and colored regions from image using HSV thresholding
  cv::Mat extract_line_regions(const cv::Mat& img);

  // Member variables
  LineDetectorNode* node_;  // Node pointer for parameters and logging
  std::vector<cv::Point2d> tracked_points_;
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

  // Line merger for combining similar lines
  std::unique_ptr<LineMerger> line_merger_;
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

  // Initialize line merger
  line_merger_ = std::make_unique<LineMerger>();
}

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

  // Line merger parameters
  bool merger_enabled = node_->declare_parameter<bool>("merger_enabled", false);
  std::string merger_method = node_->declare_parameter<std::string>(
      "merger_method", "direction_endpoint");
  double merger_max_angle =
      node_->declare_parameter<double>("merger_max_angle_diff", 20.0);
  double merger_max_endpoint_dist =
      node_->declare_parameter<double>("merger_max_endpoint_dist", 50.0);
  double merger_min_line_length =
      node_->declare_parameter<double>("merger_min_line_length", 30.0);
  int merger_prediction_frames =
      node_->declare_parameter<int>("merger_prediction_frames", 5);
  double merger_trajectory_threshold =
      node_->declare_parameter<double>("merger_trajectory_threshold", 50.0);
  double merger_confidence =
      node_->declare_parameter<double>("merger_confidence", 0.7);
  bool merger_debug = node_->declare_parameter<bool>("merger_debug", false);

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

  // Configure line merger with parameters
  if (line_merger_) {
    LineMerger::MergeConfig merge_config;
    merge_config.enabled = merger_enabled;
    merge_config.method = (merger_method == "kalman_graph")
                              ? LineMerger::MergeMethod::KALMAN_GRAPH
                              : LineMerger::MergeMethod::DIRECTION_ENDPOINT;
    merge_config.max_angle_diff = merger_max_angle;
    merge_config.max_endpoint_dist = merger_max_endpoint_dist;
    merge_config.min_line_length = merger_min_line_length;
    merge_config.prediction_frames = merger_prediction_frames;
    merge_config.trajectory_threshold = merger_trajectory_threshold;
    merge_config.merge_confidence = merger_confidence;
    merge_config.debug_enabled = merger_debug;
    line_merger_->set_config(merge_config);
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

  // Line merger parameter updates
  if (line_merger_) {
    LineMerger::MergeConfig config = line_merger_->get_config();
    bool updated = false;

    if (name == "merger_enabled") {
      config.enabled = param.as_bool();
      updated = true;
    } else if (name == "merger_method") {
      std::string method = param.as_string();
      config.method = (method == "kalman_graph")
                          ? LineMerger::MergeMethod::KALMAN_GRAPH
                          : LineMerger::MergeMethod::DIRECTION_ENDPOINT;
      updated = true;
    } else if (name == "merger_max_angle_diff") {
      config.max_angle_diff = param.as_double();
      updated = true;
    } else if (name == "merger_max_endpoint_dist") {
      config.max_endpoint_dist = param.as_double();
      updated = true;
    } else if (name == "merger_min_line_length") {
      config.min_line_length = param.as_double();
      updated = true;
    } else if (name == "merger_prediction_frames") {
      config.prediction_frames = param.as_int();
      updated = true;
    } else if (name == "merger_trajectory_threshold") {
      config.trajectory_threshold = param.as_double();
      updated = true;
    } else if (name == "merger_confidence") {
      config.merge_confidence = param.as_double();
      updated = true;
    } else if (name == "merger_debug") {
      config.debug_enabled = param.as_bool();
      updated = true;
    }

    if (updated) {
      line_merger_->set_config(config);
      return true;
    }
  }

  return false;
}

bool AdaptiveLineTracker::Impl::process_frame(
    const cv::Mat& img, AdaptiveLineTracker::DetectionResult& result) {
  tracked_points_.clear();

  // Clear result and last result
  result = AdaptiveLineTracker::DetectionResult();
  last_result_ = AdaptiveLineTracker::DetectionResult();

  if (img.empty()) {
    return false;
  }

  // Step 1: Extract ROI and convert to black mask
  cv::Rect roi_rect = valid_roi(img);
  cv::Mat work_img = img(roi_rect).clone();

  // Extract line regions (black and blue) and store for visualization
  last_black_mask_ = extract_line_regions(work_img);
  cv::Mat& black_mask =
      last_black_mask_;  // Keep variable name for compatibility

  if (black_mask.empty()) {
    return false;
  }

  // Step 2: Find contours in the mask for connectivity checking
  std::vector<std::vector<cv::Point>> all_contours;
  cv::Mat mask_copy = black_mask.clone();  // findContours modifies the input
  cv::findContours(mask_copy, all_contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // Update contour tracker with new contours
  contour_tracker_->update(all_contours);

  // Get all tracked contours (not just stable ones for line detection)
  const auto& all_tracked = contour_tracker_->get_tracked_contours();

  // Log tracking information
  if (!all_tracked.empty()) {
    std::string tracking_info = "Tracking: ";
    for (const auto& [id, tracked] : all_tracked) {
      tracking_info += "ID" + std::to_string(id) +
                       "(age:" + std::to_string(tracked.age) +
                       ",miss:" + std::to_string(tracked.missed_frames) + ") ";
    }
    RCLCPP_DEBUG(node_->get_logger(), "%s", tracking_info.c_str());
  }

  // Apply line merging if enabled
  std::vector<std::vector<cv::Point>> contours;
  std::map<int, std::set<int>>
      contour_to_ids;  // Map from contour index to original tracker IDs

  if (line_merger_ && line_merger_->get_config().enabled) {
    // Get merged contours
    contours = line_merger_->merge_lines(all_tracked);
    contour_to_ids = line_merger_->get_merged_id_mapping();

    // Log merge information
    auto merge_groups = line_merger_->get_merge_groups();
    if (!merge_groups.empty()) {
      for (const auto& [group_id, members] : merge_groups) {
        if (members.size() > 1) {
          std::string merge_info = "Merged IDs: {";
          for (int id : members) {
            merge_info += std::to_string(id) + ",";
          }
          merge_info.back() = '}';
          merge_info += " -> Contour ";
          // Find which result index contains these IDs
          for (const auto& [idx, ids] : contour_to_ids) {
            if (ids == members) {
              merge_info += std::to_string(idx);
              break;
            }
          }
          RCLCPP_DEBUG(node_->get_logger(), "%s", merge_info.c_str());
        }
      }
    }
  } else {
    // Filter tracked contours by area and missed frames (original behavior)
    constexpr double MIN_CONTOUR_AREA = 20.0;
    contours.reserve(all_tracked.size());
    int idx = 0;
    for (const auto& [id, tracked] : all_tracked) {
      if (tracked.area >= MIN_CONTOUR_AREA && tracked.missed_frames == 0) {
        contours.push_back(tracked.contour);
        contour_to_ids[idx] = {id};  // Single ID for unmerged contour
        idx++;
      }
    }
  }

  // Step 3: Collect segments and calculate scores for each contour
  std::map<int, std::vector<SegmentInfo>> contour_segments;
  auto contour_scores =
      collect_and_score_contours(black_mask, contours, contour_segments);

  // If merging is enabled, combine scores for merged contours
  if (!contour_to_ids.empty() && line_merger_ &&
      line_merger_->get_config().enabled) {
    // Create a map from original ID to contour index
    std::map<int, int> id_to_contour_idx;
    for (const auto& [idx, ids] : contour_to_ids) {
      for (int id : ids) {
        id_to_contour_idx[id] = idx;
      }
    }

    // For merged contours, combine their scores
    for (const auto& [idx, ids] : contour_to_ids) {
      if (ids.size() > 1) {
        // This is a merged contour - the score should already be calculated
        // But we should log it for debugging
        auto score_it = contour_scores.find(idx);
        if (score_it != contour_scores.end()) {
          std::string merge_score_info =
              "Merged contour " + std::to_string(idx) + " (IDs: ";
          for (int id : ids) {
            merge_score_info += std::to_string(id) + "+";
          }
          if (!ids.empty()) merge_score_info.pop_back();  // Remove last '+'
          merge_score_info +=
              ") score: " + std::to_string(score_it->second.weighted_score);
          RCLCPP_DEBUG(node_->get_logger(), "%s", merge_score_info.c_str());
        }
      }
    }
  }

  // Find the highest scoring contour
  result.best_contour_id = -1;
  result.best_contour_score = 0.0;
  result.best_contour.clear();
  std::set<int> best_merged_ids;  // Store merged IDs for best contour

  for (const auto& [contour_id, score] : contour_scores) {
    if (score.weighted_score > result.best_contour_score) {
      result.best_contour_score = score.weighted_score;
      result.best_contour_id = contour_id;
      if (contour_id >= 0 && contour_id < static_cast<int>(contours.size())) {
        result.best_contour = contours[contour_id];
        // Store merged IDs if available
        auto it = contour_to_ids.find(contour_id);
        if (it != contour_to_ids.end()) {
          best_merged_ids = it->second;
        } else {
          best_merged_ids.clear();
        }
      }
    }
  }

  // Log the best contour with merged IDs
  if (result.best_contour_id >= 0 && !best_merged_ids.empty() &&
      best_merged_ids.size() > 1) {
    std::string best_info = "Best contour: ID:";
    for (int id : best_merged_ids) {
      best_info += std::to_string(id) + "+";
    }
    if (!best_merged_ids.empty()) best_info.pop_back();  // Remove last '+'
    best_info += " (score: " + std::to_string(result.best_contour_score) + ")";
    RCLCPP_DEBUG(node_->get_logger(), "%s", best_info.c_str());
  }

  // Step 4: Build TrackedLine structures using scoring
  result.tracked_lines =
      build_tracked_lines(contour_segments, contour_scores, contours);

  // Update TrackedLine IDs to show merged IDs
  if (!contour_to_ids.empty()) {
    for (auto& line : result.tracked_lines) {
      auto it = contour_to_ids.find(line.contour_id);
      if (it != contour_to_ids.end() && !it->second.empty()) {
        // Store all merged IDs
        line.merged_ids = it->second;
        // Use the first original ID as the main ID
        line.contour_id = *it->second.begin();
      }
    }
  }

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

      // Draw ID text - check if this ID is part of a merged group
      std::string text = "ID:" + std::to_string(id);

      // Check if this ID is merged with others
      bool is_merged = false;
      std::set<int> merged_ids;
      if (line_merger_ && line_merger_->get_config().enabled) {
        auto merge_groups = line_merger_->get_merge_groups();
        for (const auto& [group_id, members] : merge_groups) {
          if (members.size() > 1 && members.find(id) != members.end()) {
            // This ID is part of a merged group - show all IDs
            text = "ID:";
            bool first = true;
            for (int mid : members) {
              if (!first) text += "+";
              text += std::to_string(mid);
              first = false;
            }
            is_merged = true;
            merged_ids = members;
            break;
          }
        }
      }

      if (tracked_contour.age > 1 && !is_merged) {
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
