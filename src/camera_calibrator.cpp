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

static cv::Mat create_hsv_mask(const cv::Mat& bgr_img, int hsv_v_min,
                               int hsv_s_max, int hsv_v_max) {
  cv::Mat hsv;
  cv::cvtColor(bgr_img, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask;
  cv::inRange(hsv, cv::Scalar(0, 0, hsv_v_min),
              cv::Scalar(180, hsv_s_max, hsv_v_max), mask);

  cv::medianBlur(mask, mask, 5);
  cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 2);

  return mask;
}

static double score_ellipse(const cv::RotatedRect& ellipse,
                            const cv::Mat& bgr_img, const cv::Mat& hsv_img,
                            int hsv_v_min, int hsv_v_max, int hsv_s_max) {
  auto work_center = cv::Point2f(static_cast<float>(bgr_img.cols) / 2.0f,
                                 static_cast<float>(bgr_img.rows) / 2.0f);

  double width = ellipse.size.width, height = ellipse.size.height;
  double angle_deg = ellipse.angle;
  if (width < height) {
    std::swap(width, height);
    angle_deg += 90.0;
  }
  while (angle_deg < 0.0) angle_deg += 180.0;
  while (angle_deg >= 180.0) angle_deg -= 180.0;

  double major = width / 2.0;

  cv::Mat ell_mask = cv::Mat::zeros(bgr_img.size(), CV_8UC1);
  cv::ellipse(ell_mask, ellipse, cv::Scalar(255), -1);
  cv::Scalar mean_hsv = cv::mean(hsv_img, ell_mask);
  double mean_s = mean_hsv[1];
  double mean_v = mean_hsv[2];

  double v_target =
      0.5 * (static_cast<double>(hsv_v_min) + static_cast<double>(hsv_v_max));

  double d = cv::norm(ellipse.center - work_center);
  double angle_pen = std::min(std::abs(angle_deg), std::abs(180.0 - angle_deg));

  double score = 0.0;
  score += d;
  score += 0.05 * major;
  score += 0.2 * angle_pen;
  score += 0.10 * std::max(0.0, mean_s - static_cast<double>(hsv_s_max));
  score += 0.002 * std::abs(mean_v - v_target);

  return score;
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
    last_circle_valid_ = false;
    last_ellipse_valid_ = false;
    last_mean_s_ = 0.0;
    last_mean_v_ = 0.0;
    last_ratio_ = 0.0;
    last_angle_deg_ = 0.0;
    last_fill_ = 0.0;
  }

  // Public interface methods
  void declare_parameters();
  bool try_update_parameter(const rclcpp::Parameter& param);
  void set_process_frame(const cv::Mat& img) { current_frame_ = img; }
  bool process_frame();
  bool is_calibration_complete() const { return calibration_complete_; }
  double get_estimated_pitch() const { return estimated_pitch_rad_; }
  double get_camera_height() const { return camera_height_m_; }
  bool detect_landmark_in_frame(double& x_full_out, double& v_full_out);
  void draw_visualization_overlay(cv::Mat& img) const;

 private:
  // Internal utility methods
  bool detect_landmark_center(double& x_full_out, double& v_full_out);
  bool find_ellipse_from_contours(
      const std::vector<std::vector<cv::Point>>& contours,
      const cv::Mat& bgr_img, const cv::Mat& hsv_img, const cv::Mat& mask,
      cv::RotatedRect& best_ellipse);
  bool find_ellipse_edge_based(const cv::Mat& bgr_img, const cv::Mat& mask,
                               cv::RotatedRect& best_ellipse);
  void try_finalize_calibration();

  // Visualization data access (internal use)
  bool has_valid_circle() const { return last_circle_valid_; }
  bool has_valid_ellipse() const { return last_ellipse_valid_; }
  cv::Point2d get_last_circle() const { return last_circle_px_; }
  cv::RotatedRect get_last_ellipse() const { return last_ellipse_full_; }
  cv::Mat get_last_hsv_mask() const { return last_calib_hsv_mask_; }
  double get_last_angle_deg() const { return last_angle_deg_; }
  double get_last_ratio() const { return last_ratio_; }
  double get_last_mean_s() const { return last_mean_s_; }
  double get_last_mean_v() const { return last_mean_v_; }

  LineDetectorNode* node_;

  // Current frame being processed
  cv::Mat current_frame_;

  // Calibration state
  bool calibration_complete_{false};
  bool calib_started_{false};
  rclcpp::Time calib_start_time_;
  double estimated_pitch_rad_{std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> v_samples_;
  static constexpr size_t kMinCalibSamples = 10;
  static constexpr size_t kMaxCalibSamples = 30;

  // Detection results for visualization
  bool last_circle_valid_{false};
  cv::Point2d last_circle_px_{};
  bool last_ellipse_valid_{false};
  cv::RotatedRect last_ellipse_full_{};
  double last_angle_deg_{};
  double last_ratio_{};
  double last_mean_s_{};
  double last_mean_v_{};
  double last_fill_{};
  cv::Mat last_calib_hsv_mask_;

  // Calibration parameters (owned by this class)
  double camera_height_m_;
  double landmark_distance_m_;
  double calib_timeout_sec_;
  std::vector<int64_t> calib_roi_;
  int calib_hsv_s_max_;
  int calib_hsv_v_min_;
  int calib_hsv_v_max_;
  int calib_min_area_;
  int calib_min_major_px_;
  double calib_max_major_ratio_;
  double calib_fill_min_;
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

void CameraCalibrator::set_process_frame(const cv::Mat& img) {
  pimpl_->set_process_frame(img);
}

bool CameraCalibrator::process_frame() { return pimpl_->process_frame(); }

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

bool CameraCalibrator::detect_landmark_in_frame(double& x_full_out,
                                                double& v_full_out) {
  return pimpl_->detect_landmark_in_frame(x_full_out, v_full_out);
}

// Implementation of Impl methods
void CameraCalibrator::Impl::declare_parameters() {
  // Calibration parameters
  camera_height_m_ =
      node_->declare_parameter<double>("camera_height_meters", 0.2);
  landmark_distance_m_ =
      node_->declare_parameter<double>("landmark_distance_meters", 0.59);
  calib_timeout_sec_ =
      node_->declare_parameter<double>("calib_timeout_sec", 60.0);
  calib_roi_ = node_->declare_parameter<std::vector<int64_t>>(
      "calib_roi", std::vector<int64_t>{200, 150, 240, 180});
  calib_hsv_s_max_ = node_->declare_parameter<int>("calib_hsv_s_max", 16);
  calib_hsv_v_min_ = node_->declare_parameter<int>("calib_hsv_v_min", 100);
  calib_hsv_v_max_ = node_->declare_parameter<int>("calib_hsv_v_max", 168);
  calib_min_area_ = node_->declare_parameter<int>("calib_min_area", 80);
  calib_min_major_px_ = node_->declare_parameter<int>("calib_min_major_px", 8);
  calib_max_major_ratio_ =
      node_->declare_parameter<double>("calib_max_major_ratio", 0.65);
  calib_fill_min_ = node_->declare_parameter<double>("calib_fill_min", 0.25);

  // Sanitize calibration parameters
  camera_height_m_ = std::max(0.01, camera_height_m_);
  landmark_distance_m_ = std::max(0.01, landmark_distance_m_);
  calib_timeout_sec_ = std::max(0.0, calib_timeout_sec_);
  calib_hsv_s_max_ = std::max(0, std::min(255, calib_hsv_s_max_));
  calib_hsv_v_min_ = std::max(0, std::min(255, calib_hsv_v_min_));
  calib_hsv_v_max_ =
      std::max(calib_hsv_v_min_, std::min(255, calib_hsv_v_max_));
  calib_min_area_ = std::max(1, calib_min_area_);
  calib_min_major_px_ = std::max(1, calib_min_major_px_);
  calib_max_major_ratio_ = std::max(0.1, std::min(1.0, calib_max_major_ratio_));
  calib_fill_min_ = std::max(0.0, std::min(1.0, calib_fill_min_));
}

bool CameraCalibrator::Impl::try_update_parameter(
    const rclcpp::Parameter& param) {
  const std::string& name = param.get_name();

  // Check if this parameter belongs to CameraCalibrator
  if (name == "camera_height_meters") {
    camera_height_m_ = std::max(0.01, param.as_double());
    return true;
  } else if (name == "landmark_distance_meters") {
    landmark_distance_m_ = std::max(0.01, param.as_double());
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
  } else if (name == "calib_min_major_px") {
    calib_min_major_px_ = std::max(1, static_cast<int>(param.as_int()));
    return true;
  } else if (name == "calib_max_major_ratio") {
    calib_max_major_ratio_ = std::max(0.1, std::min(1.0, param.as_double()));
    return true;
  } else if (name == "calib_fill_min") {
    calib_fill_min_ = std::max(0.0, std::min(1.0, param.as_double()));
    return true;
  }

  // Not a CameraCalibrator parameter
  return false;
}

bool CameraCalibrator::Impl::process_frame() {
  if (calibration_complete_) {
    return true;  // Already completed
  }

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
      return true;
    }
  }

  // Detect landmark center (ROI is applied internally)
  double x_full_out, v_full_out;
  if (detect_landmark_center(x_full_out, v_full_out)) {
    // Valid detection: store v coordinate
    v_samples_.push_back(v_full_out);
    RCLCPP_INFO(node_->get_logger(),
                "Landmark detected at (%.1f, %.1f). Samples: %zu/%zu",
                x_full_out, v_full_out, v_samples_.size(), kMinCalibSamples);

    // Store circle/ellipse data for visualization
    last_circle_px_ = cv::Point2d(x_full_out, v_full_out);
    last_circle_valid_ = true;

    // Try to finalize calibration
    try_finalize_calibration();
  } else {
    RCLCPP_DEBUG(node_->get_logger(), "No landmark detected in this frame");
  }

  return calibration_complete_;
}

bool CameraCalibrator::Impl::find_ellipse_from_contours(
    const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& bgr_img,
    const cv::Mat& hsv_img, const cv::Mat& mask,
    cv::RotatedRect& best_ellipse) {
  const double max_major =
      calib_max_major_ratio_ *
      static_cast<double>(std::min(bgr_img.cols, bgr_img.rows));

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

    double width = e.size.width, height = e.size.height;
    double angle_deg = e.angle;
    if (width < height) {
      std::swap(width, height);
      angle_deg += 90.0;
    }

    double major = width / 2.0;
    double minor = height / 2.0;
    if (major <= 0.0 || minor <= 0.0) continue;
    if (major < static_cast<double>(calib_min_major_px_)) continue;
    if (major > max_major) continue;

    double ratio = minor / major;
    if (ratio < 0.2 || ratio > 1.25) continue;

    cv::Mat ell_mask = cv::Mat::zeros(bgr_img.size(), CV_8UC1);
    cv::ellipse(ell_mask, e, cv::Scalar(255), -1);
    cv::Scalar mean_hsv = cv::mean(hsv_img, ell_mask);
    double mean_v = mean_hsv[2];

    if (mean_v < static_cast<double>(calib_hsv_v_min_) + 10.0) continue;

    cv::Mat mask_inside;
    cv::bitwise_and(mask, ell_mask, mask_inside);
    const double ellipse_area = CV_PI * major * minor;
    const double fill =
        ellipse_area > 1.0
            ? static_cast<double>(cv::countNonZero(mask_inside)) / ellipse_area
            : 0.0;

    if (fill < calib_fill_min_) continue;

    double score = score_ellipse(e, bgr_img, hsv_img, calib_hsv_v_min_,
                                 calib_hsv_v_max_, calib_hsv_s_max_);

    if (score < best_score) {
      best_score = score;
      best_ellipse = e;
      found = true;
    }
  }

  return found;
}

bool CameraCalibrator::Impl::find_ellipse_edge_based(
    const cv::Mat& bgr_img, const cv::Mat& mask,
    cv::RotatedRect& best_ellipse) {
  RCLCPP_DEBUG(node_->get_logger(),
               "No landmark found in contours, trying edge-based detection");

  cv::Mat gray;
  cv::cvtColor(bgr_img, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.2);
  cv::Mat edges;
  cv::Canny(gray, edges, 60, 180, 3, true);
  cv::bitwise_and(edges, mask, edges);
  cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::dilate(edges, edges, kd);

  std::vector<std::vector<cv::Point>> ctr2;
  cv::findContours(edges, ctr2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  auto work_center = cv::Point2f(static_cast<float>(bgr_img.cols) / 2.0f,
                                 static_cast<float>(bgr_img.rows) / 2.0f);
  const double max_major =
      calib_max_major_ratio_ *
      static_cast<double>(std::min(bgr_img.cols, bgr_img.rows));

  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();

  for (const auto& cnt : ctr2) {
    if (cnt.size() < 20) continue;
    double area2 = cv::contourArea(cnt);
    if (area2 < 60.0) continue;

    if (cnt.size() >= 5) {
      cv::RotatedRect e = cv::fitEllipse(cnt);
      double major = std::max(e.size.width, e.size.height) / 2.0;
      double minor = std::min(e.size.width, e.size.height) / 2.0;
      if (major <= 0.0 || minor <= 0.0) continue;
      if (major > max_major) continue;

      double ratio2 = minor / major;
      if (ratio2 < 0.2 || ratio2 > 1.3) continue;

      double d = cv::norm(e.center - work_center);
      double score = d + 0.12 * major;

      if (score < best_score) {
        best_score = score;
        best_ellipse = e;
        found = true;
      }
    }
  }

  return found;
}

bool CameraCalibrator::Impl::detect_landmark_center(double& x_full_out,
                                                    double& v_full_out) {
  if (current_frame_.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty image in detect_landmark_center");
    return false;
  }

  // Apply ROI internally
  cv::Rect roi = valid_roi(current_frame_, calib_roi_);
  cv::Mat work_img = current_frame_(roi).clone();
  const double scale = 1.0;  // No scaling currently

  // Convert to BGR if needed
  cv::Mat work_bgr;
  if (work_img.channels() == 1) {
    cv::cvtColor(work_img, work_bgr, cv::COLOR_GRAY2BGR);
  } else {
    work_bgr = work_img;
  }

  // Create HSV mask for gray landmark detection
  cv::Mat hsv;
  cv::cvtColor(work_bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask = create_hsv_mask(work_bgr, calib_hsv_v_min_, calib_hsv_s_max_,
                                 calib_hsv_v_max_);

  // Store processed HSV mask for debug visualization
  last_calib_hsv_mask_ = mask.clone();

  // Find contours in the mask
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  RCLCPP_DEBUG(node_->get_logger(), "Found %zu contours in HSV mask",
               contours.size());

  // Try to find best ellipse from contours
  cv::RotatedRect best_ellipse;
  bool found =
      find_ellipse_from_contours(contours, work_bgr, hsv, mask, best_ellipse);

  // If no ellipse found, try edge-based detection as fallback
  if (!found) {
    found = find_ellipse_edge_based(work_bgr, mask, best_ellipse);
  }

  if (!found) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "No landmark found after edge-based detection");
    return false;
  }

  // Compute final metrics for the selected ellipse
  double sel_width = best_ellipse.size.width;
  double sel_height = best_ellipse.size.height;
  double sel_angle_deg = best_ellipse.angle;
  if (sel_width < sel_height) {
    std::swap(sel_width, sel_height);
    sel_angle_deg += 90.0;
  }
  while (sel_angle_deg < 0.0) sel_angle_deg += 180.0;
  while (sel_angle_deg >= 180.0) sel_angle_deg -= 180.0;

  double sel_major = sel_width / 2.0;
  double sel_minor = sel_height / 2.0;
  cv::Mat sel_mask = cv::Mat::zeros(work_bgr.size(), CV_8UC1);
  cv::ellipse(sel_mask, best_ellipse, cv::Scalar(255), -1);
  cv::Scalar sel_mean_hsv = cv::mean(hsv, sel_mask);

  // Store metrics for visualization
  last_mean_s_ = sel_mean_hsv[1];
  last_mean_v_ = sel_mean_hsv[2];
  last_ratio_ = (sel_major > 1e-9) ? (sel_minor / sel_major) : 0.0;
  last_angle_deg_ = sel_angle_deg;

  const double sel_area = CV_PI * sel_major * sel_minor;
  if (sel_area > 1.0) {
    cv::Mat mask_inside;
    cv::bitwise_and(mask, sel_mask, mask_inside);
    last_fill_ = static_cast<double>(cv::countNonZero(mask_inside)) / sel_area;
  } else {
    last_fill_ = 0.0;
  }

  // Map to full image coordinates
  const double x_full =
      static_cast<double>(best_ellipse.center.x) * scale + roi.x;
  const double v_full =
      static_cast<double>(best_ellipse.center.y) * scale + roi.y;
  x_full_out = x_full;
  v_full_out = v_full;

  // Store last ellipse in full-res coordinates for overlay
  last_ellipse_full_ = cv::RotatedRect(
      cv::Point2f(static_cast<float>(x_full), static_cast<float>(v_full)),
      cv::Size2f(static_cast<float>(best_ellipse.size.width * scale),
                 static_cast<float>(best_ellipse.size.height * scale)),
      best_ellipse.angle);
  last_ellipse_valid_ = true;

  return true;
}

bool CameraCalibrator::Impl::detect_landmark_in_frame(double& x_full_out,
                                                      double& v_full_out) {
  // Detect landmark in current frame
  return detect_landmark_center(x_full_out, v_full_out);
}

void CameraCalibrator::Impl::try_finalize_calibration() {
  // Timeout check
  if (calib_started_ && calib_timeout_sec_ > 0.0) {
    const rclcpp::Duration elapsed = node_->now() - calib_start_time_;
    if (elapsed.seconds() >= calib_timeout_sec_) {
      RCLCPP_WARN(node_->get_logger(),
                  "Calibration timeout after %.1fs. Proceed without pitch.",
                  calib_timeout_sec_);
      estimated_pitch_rad_ = 0.0;
      calibration_complete_ = true;
      return;
    }
  }

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
  const double D = landmark_distance_m_;
  const double h = camera_height_m_;
  // tan(theta) = (D*u - h) / (D + h*u)
  const double denom = (D + h * u);
  if (std::abs(denom) < 1e-6) return;  // avoid singularities

  const double t = (D * u - h) / denom;
  const double theta = std::atan(t);

  // Sanity clamp to [-45, 45] deg
  const double max_rad = deg2rad(45.0);
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
  if (!has_valid_ellipse()) {
    return;
  }

  // Draw the detected ellipse in cyan
  cv::ellipse(img, last_ellipse_full_, cv::Scalar(0, 255, 255), 2);

  // Draw the center point as a red filled circle
  if (last_circle_valid_) {
    cv::circle(img,
               cv::Point(static_cast<int>(last_circle_px_.x),
                         static_cast<int>(last_circle_px_.y)),
               5, cv::Scalar(0, 0, 255), -1);
  }

  // Add calibration info text showing S and V values
  cv::putText(img,
              cv::format("Calib: S=%.1f V=%.1f", last_mean_s_, last_mean_v_),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 255), 1);
}
