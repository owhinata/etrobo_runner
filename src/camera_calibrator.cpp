#include "src/camera_calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "src/line_detector_node.hpp"

// Small helpers for angle conversion
static inline double deg2rad(double deg) { return deg * CV_PI / 180.0; }
static inline double rad2deg(double rad) { return rad * 180.0 / CV_PI; }

// Helper function for median calculation
static double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const size_t n = v.size();
  std::nth_element(v.begin(), v.begin() + n / 2, v.end());
  double m = v[n / 2];
  if ((n % 2) == 0) {
    auto max_it = std::max_element(v.begin(), v.begin() + n / 2);
    m = (m + *max_it) * 0.5;
  }
  return m;
}

// Static helper functions for image processing
static cv::Rect valid_roi(const cv::Mat& img, const std::vector<int64_t>& roi) {
  if (roi.size() != 4) return cv::Rect(0, 0, img.cols, img.rows);
  int x = static_cast<int>(roi[0]);
  int y = static_cast<int>(roi[1]);
  int w = static_cast<int>(roi[2]);
  int h = static_cast<int>(roi[3]);
  if (x < 0 || y < 0 || w <= 0 || h <= 0) {
    return cv::Rect(0, 0, img.cols, img.rows);
  }
  x = std::max(0, std::min(x, img.cols - 1));
  y = std::max(0, std::min(y, img.rows - 1));
  w = std::min(w, img.cols - x);
  h = std::min(h, img.rows - y);
  return cv::Rect(x, y, w, h);
}

// Implementation class definition
class CameraCalibrator::Impl {
 public:
  explicit Impl(LineDetectorNode* node) : node_(node) {
    // Initialize calibration state
    calib_started_ = false;
    calibration_complete_ = false;
    estimated_pitch_rad_ = std::numeric_limits<double>::quiet_NaN();

    // Landmark detection state
    last_ellipse_valid_ = false;
    last_mean_s_ = 0.0;
    last_mean_v_ = 0.0;
    last_ratio_ = 0.0;
    last_angle_deg_ = 0.0;
  }

  // Public interface methods
  void declare_parameters();
  bool try_update_parameter(const rclcpp::Parameter& param);
  bool process_frame(const cv::Mat& img, cv::Point2d& landmark_pos);
  bool is_calibration_complete() const { return calibration_complete_; }
  double get_estimated_pitch() const { return estimated_pitch_rad_; }
  double get_camera_height() const { return camera_height_meters_; }
  void draw_visualization_overlay(cv::Mat& img) const;

 private:
  // Internal utility methods
  bool detect_landmark_center(const cv::Mat& img, cv::Point2d& landmark_pos);
  bool find_ellipse_from_contours(
      const std::vector<std::vector<cv::Point>>& contours,
      const cv::Mat& bgr_img, const cv::Mat& hsv_img, const cv::Mat& mask,
      cv::RotatedRect& best_ellipse);
  void try_finalize_calibration();

  // Visualization data access (internal use)
  bool has_valid_ellipse() const { return last_ellipse_valid_; }

  LineDetectorNode* node_;

  // Calibration state
  bool calibration_complete_{false};
  bool calib_started_{false};
  rclcpp::Time calib_start_time_;
  double estimated_pitch_rad_{std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> v_samples_;
  static constexpr size_t kMinCalibSamples = 10;
  static constexpr size_t kMaxCalibSamples = 30;

  // Detection thresholds
  static constexpr double kMinFillRatio =
      0.6;  // Minimum fill ratio for valid detection
  static constexpr double kMinAspectRatio =
      0.2;  // Minimum aspect ratio (minor/major) to filter noise

  // Detection results for visualization
  cv::Point2d last_circle_px_{};
  bool last_ellipse_valid_{false};
  cv::RotatedRect last_ellipse_full_{};
  double last_angle_deg_{};
  double last_ratio_{};
  double last_mean_s_{};
  double last_mean_v_{};
  cv::Mat last_calib_hsv_mask_;

  // Calibration parameters (owned by this class)
  double camera_height_meters_;
  double landmark_distance_meters_;
  double calib_timeout_sec_;
  std::vector<int64_t> calib_roi_;
  int calib_hsv_s_max_;
  int calib_hsv_v_min_;
  int calib_hsv_v_max_;
  int calib_min_area_;
};

// CameraCalibrator public interface implementation
CameraCalibrator::CameraCalibrator(LineDetectorNode* node)
    : pimpl_(std::make_unique<Impl>(node)) {
  declare_parameters();
}

CameraCalibrator::~CameraCalibrator() = default;

void CameraCalibrator::declare_parameters() { pimpl_->declare_parameters(); }

bool CameraCalibrator::try_update_parameter(const rclcpp::Parameter& param) {
  return pimpl_->try_update_parameter(param);
}

bool CameraCalibrator::process_frame(const cv::Mat& img,
                                     cv::Point2d& landmark_pos) {
  return pimpl_->process_frame(img, landmark_pos);
}

bool CameraCalibrator::is_calibration_complete() const {
  return pimpl_->is_calibration_complete();
}

double CameraCalibrator::get_estimated_pitch() const {
  return pimpl_->get_estimated_pitch();
}

double CameraCalibrator::get_camera_height() const {
  return pimpl_->get_camera_height();
}

void CameraCalibrator::draw_visualization_overlay(cv::Mat& img) const {
  pimpl_->draw_visualization_overlay(img);
}

// Implementation of Impl methods
void CameraCalibrator::Impl::declare_parameters() {
  // Calibration parameters
  camera_height_meters_ =
      node_->declare_parameter<double>("camera_height_meters", 0.2);
  landmark_distance_meters_ =
      node_->declare_parameter<double>("landmark_distance_meters", 0.59);
  calib_timeout_sec_ =
      node_->declare_parameter<double>("calib_timeout_sec", 60.0);
  calib_roi_ = node_->declare_parameter<std::vector<int64_t>>(
      "calib_roi", std::vector<int64_t>{200, 150, 240, 180});
  calib_hsv_s_max_ = node_->declare_parameter<int>("calib_hsv_s_max", 16);
  calib_hsv_v_min_ = node_->declare_parameter<int>("calib_hsv_v_min", 100);
  calib_hsv_v_max_ = node_->declare_parameter<int>("calib_hsv_v_max", 168);
  calib_min_area_ = node_->declare_parameter<int>("calib_min_area", 80);

  // Sanitize calibration parameters
  camera_height_meters_ = std::max(0.01, camera_height_meters_);
  landmark_distance_meters_ = std::max(0.01, landmark_distance_meters_);
  calib_timeout_sec_ = std::max(0.0, calib_timeout_sec_);
  calib_hsv_s_max_ = std::max(0, std::min(255, calib_hsv_s_max_));
  calib_hsv_v_min_ = std::max(0, std::min(255, calib_hsv_v_min_));
  calib_hsv_v_max_ =
      std::max(calib_hsv_v_min_, std::min(255, calib_hsv_v_max_));
  calib_min_area_ = std::max(1, calib_min_area_);
}

bool CameraCalibrator::Impl::try_update_parameter(
    const rclcpp::Parameter& param) {
  const std::string& name = param.get_name();

  // Check if this parameter belongs to CameraCalibrator
  if (name == "camera_height_meters") {
    camera_height_meters_ = std::max(0.01, param.as_double());
    return true;
  } else if (name == "landmark_distance_meters") {
    landmark_distance_meters_ = std::max(0.01, param.as_double());
    return true;
  } else if (name == "calib_timeout_sec") {
    calib_timeout_sec_ = std::max(0.0, param.as_double());
    return true;
  } else if (name == "calib_roi") {
    calib_roi_ = param.as_integer_array();
    return true;
  } else if (name == "calib_hsv_s_max") {
    calib_hsv_s_max_ =
        std::max(0, std::min(255, static_cast<int>(param.as_int())));
    return true;
  } else if (name == "calib_hsv_v_min") {
    calib_hsv_v_min_ =
        std::max(0, std::min(255, static_cast<int>(param.as_int())));
    return true;
  } else if (name == "calib_hsv_v_max") {
    calib_hsv_v_max_ = std::max(
        calib_hsv_v_min_, std::min(255, static_cast<int>(param.as_int())));
    return true;
  } else if (name == "calib_min_area") {
    calib_min_area_ = std::max(1, static_cast<int>(param.as_int()));
    return true;
  }

  // Not a CameraCalibrator parameter
  return false;
}

bool CameraCalibrator::Impl::process_frame(const cv::Mat& img,
                                           cv::Point2d& landmark_pos) {
  if (!calibration_complete_) {
    // Initialize calibration start time if not started
    if (!calib_started_) {
      calib_started_ = true;
      calib_start_time_ = node_->now();
      RCLCPP_INFO(node_->get_logger(), "Calibration started");
    }

    // Check for timeout
    if (calib_timeout_sec_ > 0.0) {
      auto elapsed = (node_->now() - calib_start_time_).seconds();
      if (elapsed > calib_timeout_sec_) {
        RCLCPP_WARN(node_->get_logger(),
                    "Calibration timeout (%.1f sec). Skipping calibration.",
                    calib_timeout_sec_);
        calibration_complete_ = true;
      }
    }
  }

  // Detect landmark center (ROI is applied internally)
  if (detect_landmark_center(img, landmark_pos)) {
    if (!calibration_complete_) {
      // Valid detection: store v coordinate
      v_samples_.push_back(landmark_pos.y);
      RCLCPP_INFO(node_->get_logger(),
                  "Landmark detected at (%.1f, %.1f). Samples: %zu/%zu",
                  landmark_pos.x, landmark_pos.y, v_samples_.size(),
                  kMinCalibSamples);

      // Store circle/ellipse data for visualization
      last_circle_px_ = landmark_pos;

      // Try to finalize calibration
      try_finalize_calibration();
    }
  } else {
    RCLCPP_DEBUG(node_->get_logger(), "No landmark detected in this frame");
    return false;
  }

  return true;
}

bool CameraCalibrator::Impl::find_ellipse_from_contours(
    const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& bgr_img,
    const cv::Mat& hsv_img, const cv::Mat& mask,
    cv::RotatedRect& best_ellipse) {
  const double max_major =
      static_cast<double>(std::min(bgr_img.cols, bgr_img.rows));

  // Pre-calculate v_target once
  const double v_target = 0.5 * (static_cast<double>(calib_hsv_v_min_) +
                                 static_cast<double>(calib_hsv_v_max_));

  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();

  for (const auto& cnt : contours) {
    if (cnt.size() < 5) continue;
    double area = cv::contourArea(cnt);
    if (area < static_cast<double>(calib_min_area_)) {
      RCLCPP_DEBUG(node_->get_logger(), "Contour rejected: area %.1f < min %d",
                   area, calib_min_area_);
      continue;
    }

    cv::RotatedRect e = cv::fitEllipse(cnt);

    // Calculate major/minor axes and normalize angle
    double width = e.size.width, height = e.size.height;
    double angle_deg = e.angle;
    if (width < height) {
      std::swap(width, height);
      angle_deg += 90.0;
    }

    double major = width / 2.0;
    double minor = height / 2.0;
    if (major <= 0.0 || minor <= 0.0) continue;
    if (major > max_major) continue;

    double ratio = minor / major;
    if (ratio < kMinAspectRatio) continue;  // Too elongated, likely noise

    // Create ellipse mask and calculate HSV statistics
    cv::Mat ell_mask = cv::Mat::zeros(bgr_img.size(), CV_8UC1);
    cv::ellipse(ell_mask, e, cv::Scalar(255), -1);
    cv::Scalar mean_hsv = cv::mean(hsv_img, ell_mask);
    double mean_s = mean_hsv[1];
    double mean_v = mean_hsv[2];

    // Calculate fill ratio
    cv::Mat mask_inside;
    cv::bitwise_and(mask, ell_mask, mask_inside);
    const double ellipse_area = CV_PI * major * minor;
    const double fill =
        ellipse_area > 1.0
            ? static_cast<double>(cv::countNonZero(mask_inside)) / ellipse_area
            : 0.0;

    if (fill < kMinFillRatio) continue;  // Fixed threshold for fill ratio

    // Calculate score
    // Normalize angle to [-90, 90]
    while (angle_deg > 90.0) angle_deg -= 180.0;
    while (angle_deg <= -90.0) angle_deg += 180.0;

    double angle_pen = std::abs(angle_deg);

    double score =
        0.2 * angle_pen +
        0.10 * std::max(0.0, mean_s - static_cast<double>(calib_hsv_s_max_)) +
        0.002 * std::abs(mean_v - v_target);

    if (score < best_score) {
      best_score = score;
      best_ellipse = e;
      // Store metrics for visualization
      last_angle_deg_ = angle_deg;
      last_ratio_ = ratio;
      last_mean_s_ = mean_s;
      last_mean_v_ = mean_v;
      found = true;
    }
  }

  return found;
}

bool CameraCalibrator::Impl::detect_landmark_center(const cv::Mat& img,
                                                    cv::Point2d& landmark_pos) {
  // Reset detection flag at the beginning of each detection attempt
  last_ellipse_valid_ = false;

  if (img.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty image in detect_landmark_center");
    return false;
  }

  // Apply ROI and prepare BGR image
  cv::Rect roi = valid_roi(img, calib_roi_);
  cv::Mat roi_img = img(roi);  // No clone needed, just a view
  const double scale = 1.0;    // No scaling currently

  cv::Mat work_bgr;
  if (roi_img.channels() == 1) {
    cv::cvtColor(roi_img, work_bgr, cv::COLOR_GRAY2BGR);
  } else if (roi_img.channels() == 3) {
    work_bgr = roi_img;  // Direct use of ROI view
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Unexpected image channels: %d",
                 roi_img.channels());
    return false;
  }

  // Convert to HSV and create mask for gray landmark detection
  cv::Mat hsv;
  cv::cvtColor(work_bgr, hsv, cv::COLOR_BGR2HSV);

  // Use reference to last_calib_hsv_mask_ to avoid copy
  cv::Mat& mask = last_calib_hsv_mask_;
  cv::inRange(hsv, cv::Scalar(0, 0, calib_hsv_v_min_),
              cv::Scalar(180, calib_hsv_s_max_, calib_hsv_v_max_), mask);

  // Apply noise removal and morphological operations:
  // - Median filter (5x5) to remove salt-and-pepper noise
  // - Opening to remove small noise regions
  // - Closing x2 to fill holes and smooth boundaries
  cv::medianBlur(mask, mask, 5);
  cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 2);

  // Find best ellipse from contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  RCLCPP_DEBUG(node_->get_logger(), "Found %zu contours in HSV mask",
               contours.size());

  cv::RotatedRect best_ellipse;
  bool found =
      find_ellipse_from_contours(contours, work_bgr, hsv, mask, best_ellipse);
  if (!found) {
    RCLCPP_DEBUG(node_->get_logger(), "No landmark found in contours");
    return false;
  }

  // Map to full image coordinates
  landmark_pos.x = static_cast<double>(best_ellipse.center.x) * scale + roi.x;
  landmark_pos.y = static_cast<double>(best_ellipse.center.y) * scale + roi.y;

  // Store last ellipse in full-res coordinates for overlay
  last_ellipse_full_ = cv::RotatedRect(
      cv::Point2f(static_cast<float>(landmark_pos.x),
                  static_cast<float>(landmark_pos.y)),
      cv::Size2f(static_cast<float>(best_ellipse.size.width * scale),
                 static_cast<float>(best_ellipse.size.height * scale)),
      best_ellipse.angle);
  last_ellipse_valid_ = true;

  return true;
}

void CameraCalibrator::Impl::try_finalize_calibration() {
  if (!node_->has_cam_info_) {
    RCLCPP_DEBUG(node_->get_logger(), "Waiting for camera info...");
    return;
  }
  if (v_samples_.size() < kMinCalibSamples) {
    RCLCPP_DEBUG(node_->get_logger(), "Not enough samples: %zu < %zu",
                 v_samples_.size(), kMinCalibSamples);
    return;
  }

  const double v_med = median(v_samples_);
  // Compute u = (v - cy)/fy
  const double u = (v_med - node_->cy_) / node_->fy_;
  const double D = landmark_distance_meters_;
  const double h = camera_height_meters_;
  // φ = atan(h/D) - atan(u)
  // tan(φ) = (D*u - h) / (D + h*u)
  const double denom = (D + h * u);
  if (std::abs(denom) < 1e-6) return;  // avoid singularities

  const double t = (D * u - h) / denom;
  const double theta = std::atan(t);

  // Sanity clamp to [-90, 90] deg
  const double max_rad = deg2rad(90.0);
  const double pitch_rad = std::max(-max_rad, std::min(theta, max_rad));

  if (calib_timeout_sec_ == 0.0) {
    // Continuous calibration mode: keep updating pitch and stay in this state
    estimated_pitch_rad_ = pitch_rad;
    return;
  }

  estimated_pitch_rad_ = pitch_rad;
  calibration_complete_ = true;
  RCLCPP_INFO(node_->get_logger(),
              "Calibration complete! Estimated pitch: %.2f deg",
              rad2deg(estimated_pitch_rad_));
}

void CameraCalibrator::Impl::draw_visualization_overlay(cv::Mat& img) const {
  // Draw ROI rectangle in blue
  if (!calib_roi_.empty() && calib_roi_.size() == 4) {
    cv::Rect roi = valid_roi(img, calib_roi_);
    cv::rectangle(img, roi, cv::Scalar(255, 0, 0),
                  1);  // Blue rectangle (thinner)

    // Draw dashed crosshair lines at ROI center
    // Calculate center of ROI
    int center_x = roi.x + roi.width / 2;
    int center_y = roi.y + roi.height / 2;

    // Draw horizontal dashed line (within ROI)
    const int dash_length = 5;
    const int gap_length = 5;
    for (int x = roi.x; x < roi.x + roi.width; x += dash_length + gap_length) {
      int x_end = std::min(x + dash_length, roi.x + roi.width);
      cv::line(img, cv::Point(x, center_y), cv::Point(x_end, center_y),
               cv::Scalar(128, 128, 128), 1);  // Gray dashed line
    }

    // Draw vertical dashed line (within ROI)
    for (int y = roi.y; y < roi.y + roi.height; y += dash_length + gap_length) {
      int y_end = std::min(y + dash_length, roi.y + roi.height);
      cv::line(img, cv::Point(center_x, y), cv::Point(center_x, y_end),
               cv::Scalar(128, 128, 128), 1);  // Gray dashed line
    }
  }

  // Draw detected ellipse and landmark info if available
  if (has_valid_ellipse()) {
    // Draw the detected ellipse in yellow
    cv::ellipse(img, last_ellipse_full_, cv::Scalar(0, 255, 255), 2);

    // Draw the center point as a red filled circle
    cv::circle(img,
               cv::Point(static_cast<int>(last_circle_px_.x),
                         static_cast<int>(last_circle_px_.y)),
               5, cv::Scalar(0, 0, 255), -1);

    // Draw line from bottom center (robot position) to landmark center in cyan
    cv::Point robot_pos(img.cols / 2, img.rows - 1);  // Bottom center
    cv::Point landmark_pos(static_cast<int>(last_circle_px_.x),
                           static_cast<int>(last_circle_px_.y));
    cv::line(img, robot_pos, landmark_pos, cv::Scalar(255, 255, 0),
             2);  // Cyan line

    // Display calibration values: a (angle), r (ratio), v (HSV value), s
    // (saturation)
    std::string calib_text =
        cv::format("a=%.1f r=%.2f v=%.0f s=%.0f", last_angle_deg_, last_ratio_,
                   last_mean_v_, last_mean_s_);

    // Calculate text size for background rectangle
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(calib_text, cv::FONT_HERSHEY_SIMPLEX,
                                         0.4, 1, &baseline);

    // Draw semi-transparent background rectangle
    cv::Point text_pos(static_cast<int>(last_circle_px_.x) + 20,
                       static_cast<int>(last_circle_px_.y) - 10);
    cv::Rect text_bg(text_pos.x - 2, text_pos.y - text_size.height - 2,
                     text_size.width + 4, text_size.height + baseline + 4);

    // Create overlay for semi-transparent effect
    cv::Mat overlay;
    img.copyTo(overlay);
    cv::rectangle(overlay, text_bg, cv::Scalar(0, 0, 0),
                  -1);                                // Black filled rectangle
    cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);  // 80% transparency

    cv::putText(img, calib_text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0, 255, 255), 1);  // Yellow text (smaller font)
  }

  // Draw HSV Mask as Picture-in-Picture in top right corner
  if (!last_calib_hsv_mask_.empty()) {
    // Calculate PiP size and position (about 1/4 of the image width)
    int pip_width = img.cols / 4;
    int pip_height = img.rows / 4;
    int pip_x = img.cols - pip_width - 10;  // 10 pixels margin from right
    int pip_y = 10;                         // 10 pixels margin from top

    // Resize the HSV mask to PiP size
    cv::Mat pip_mask;
    cv::resize(last_calib_hsv_mask_, pip_mask, cv::Size(pip_width, pip_height));

    // Convert grayscale mask to BGR for overlay
    cv::Mat pip_bgr;
    cv::cvtColor(pip_mask, pip_bgr, cv::COLOR_GRAY2BGR);

    // Create ROI in the main image and copy PiP
    cv::Rect pip_roi(pip_x, pip_y, pip_width, pip_height);
    pip_bgr.copyTo(img(pip_roi));

    // Draw border around PiP
    cv::rectangle(img, pip_roi, cv::Scalar(255, 255, 255), 1);  // White border

    // Add "HSV Mask" label inside the PiP area at the top
    cv::putText(img, "HSV Mask",
                cv::Point(pip_x + 2, pip_y + 10),  // Move inside the PiP area
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255),
                1);  // White text
  }

  // Display distance and pitch info at top left
  std::string info_text =
      cv::format("D=%.2fm, pitch=%.1f deg", landmark_distance_meters_,
                 rad2deg(estimated_pitch_rad_));

  // Calculate text size for background rectangle
  int baseline = 0;
  cv::Size text_size =
      cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

  // Draw semi-transparent background rectangle
  cv::Point text_pos(10, 30);
  cv::Rect text_bg(text_pos.x - 2, text_pos.y - text_size.height - 2,
                   text_size.width + 4, text_size.height + baseline + 4);

  // Create overlay for semi-transparent effect
  cv::Mat overlay;
  img.copyTo(overlay);
  cv::rectangle(overlay, text_bg, cv::Scalar(0, 0, 0),
                -1);                                // Black filled rectangle
  cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);  // 80% transparency

  cv::putText(img, info_text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 255), 1);  // Yellow text (smaller font)
}
