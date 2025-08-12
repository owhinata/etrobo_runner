#include "src/camera_calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>

#include "src/line_detector_node.hpp"

// Small helpers for angle conversion
static inline double deg2rad(double deg) { return deg * CV_PI / 180.0; }
static inline double rad2deg(double rad) { return rad * 180.0 / CV_PI; }

CameraCalibrator::CameraCalibrator(LineDetectorNode* node) : node_(node) {
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

  // Cache parameters from node
  camera_height_m_ = node_->camera_height_m_;
  landmark_distance_m_ = node_->landmark_distance_m_;
  calib_timeout_sec_ = node_->calib_timeout_sec_;
  calib_roi_ = node_->calib_roi_;
  calib_hsv_s_max_ = node_->calib_hsv_s_max_;
  calib_hsv_v_min_ = node_->calib_hsv_v_min_;
  calib_hsv_v_max_ = node_->calib_hsv_v_max_;
  calib_min_area_ = node_->calib_min_area_;
  calib_min_major_px_ = node_->calib_min_major_px_;
  calib_max_major_ratio_ = node_->calib_max_major_ratio_;
  calib_fill_min_ = node_->calib_fill_min_;

  // Start calibration timer if timeout is enabled
  if (calib_timeout_sec_ > 0.0) {
    calib_start_time_ = node_->now();
    calib_started_ = true;
  }
}

bool CameraCalibrator::process_frame(const cv::Mat& img) {
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

  // Detect landmark center in the calibration ROI
  cv::Rect calib_rect = valid_roi(img, calib_roi_);
  RCLCPP_DEBUG(node_->get_logger(), "Calibration ROI: x=%d, y=%d, w=%d, h=%d",
               calib_rect.x, calib_rect.y, calib_rect.width, calib_rect.height);

  // Extract work region (apply scaling if needed)
  cv::Mat work = img(calib_rect).clone();
  const double scale = 1.0;  // No scaling currently

  double x_full_out, v_full_out;
  if (detect_landmark_center(work, calib_rect, scale, x_full_out, v_full_out)) {
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

cv::Rect CameraCalibrator::valid_roi(const cv::Mat& img,
                                     const std::vector<int64_t>& roi) {
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

bool CameraCalibrator::detect_landmark_center(const cv::Mat& work_img,
                                              const cv::Rect& roi, double scale,
                                              double& x_full_out,
                                              double& v_full_out) {
  if (work_img.empty()) {
    RCLCPP_WARN(node_->get_logger(),
                "Empty work image in detect_landmark_center");
    return false;
  }

  cv::Mat work_bgr;
  if (work_img.channels() == 1) {
    cv::cvtColor(work_img, work_bgr, cv::COLOR_GRAY2BGR);
  } else {
    work_bgr = work_img;
  }

  // HSV gray mask: low saturation, mid brightness
  cv::Mat hsv;
  cv::cvtColor(work_bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask;
  cv::inRange(hsv, cv::Scalar(0, 0, calib_hsv_v_min_),
              cv::Scalar(180, calib_hsv_s_max_, calib_hsv_v_max_), mask);

  cv::medianBlur(mask, mask, 5);
  cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  // Open then Close to clean noise and fill the disk
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 2);

  // Store processed HSV mask for debug visualization
  last_calib_hsv_mask_ = mask.clone();

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  RCLCPP_DEBUG(node_->get_logger(), "Found %zu contours in HSV mask",
               contours.size());

  auto work_center = cv::Point2f(static_cast<float>(work_bgr.cols) / 2.0f,
                                 static_cast<float>(work_bgr.rows) / 2.0f);
  bool found = false;
  cv::RotatedRect best_ellipse;
  double best_score = std::numeric_limits<double>::infinity();
  const double max_major =
      calib_max_major_ratio_ *
      static_cast<double>(std::min(work_bgr.cols, work_bgr.rows));

  for (const auto& cnt : contours) {
    if (cnt.size() < 5) continue;
    double area = cv::contourArea(cnt);
    if (area < static_cast<double>(calib_min_area_)) {
      RCLCPP_DEBUG(node_->get_logger(), "Contour rejected: area %.1f < min %d",
                   area, calib_min_area_);
      continue;
    }

    cv::RotatedRect e = cv::fitEllipse(cnt);

    // Normalize so 'major' is along angle
    double width = e.size.width, height = e.size.height;
    double angle_deg = e.angle;
    if (width < height) {
      std::swap(width, height);
      angle_deg += 90.0;
    }
    while (angle_deg < 0.0) angle_deg += 180.0;
    while (angle_deg >= 180.0) angle_deg -= 180.0;

    double major = width / 2.0;
    double minor = height / 2.0;
    if (major <= 0.0 || minor <= 0.0) continue;
    if (major < static_cast<double>(calib_min_major_px_)) continue;
    if (major > max_major) continue;  // reject too large objects

    double ratio = minor / major;
    if (ratio < 0.2 || ratio > 1.25) continue;  // plausible ellipse

    // Grayness inside ellipse (prefer low S, mid V)
    cv::Mat ell_mask = cv::Mat::zeros(work_bgr.size(), CV_8UC1);
    cv::ellipse(ell_mask, e, cv::Scalar(255), -1);
    cv::Scalar mean_hsv = cv::mean(hsv, ell_mask);
    double mean_s = mean_hsv[1];
    double mean_v = mean_hsv[2];

    // Hard reject if interior is too dark (likely black line)
    if (mean_v < static_cast<double>(calib_hsv_v_min_) + 10.0) continue;

    // Fill ratio: mask pixels inside ellipse vs ellipse area
    cv::Mat mask_inside;
    cv::bitwise_and(mask, ell_mask, mask_inside);
    const double ellipse_area = CV_PI * major * minor;
    const double fill =
        ellipse_area > 1.0
            ? static_cast<double>(cv::countNonZero(mask_inside)) / ellipse_area
            : 0.0;

    if (fill < calib_fill_min_) continue;

    double v_target = 0.5 * (static_cast<double>(calib_hsv_v_min_) +
                             static_cast<double>(calib_hsv_v_max_));

    // Scoring: center distance + size + angle preference + grayness penalties
    double d = cv::norm(e.center - work_center);
    double angle_pen =
        std::min(std::abs(angle_deg), std::abs(180.0 - angle_deg));
    // We prefer horizontal major axis (squashed vertically) -> angle near
    // 0/180
    double score = 0.0;
    score += d;                // center proximity
    score += 0.05 * major;     // size penalty
    score += 0.2 * angle_pen;  // orientation preference
    score +=
        0.10 * std::max(0.0, mean_s - static_cast<double>(calib_hsv_s_max_));
    score += 0.002 * std::abs(mean_v - v_target);

    if (score < best_score) {
      best_score = score;
      best_ellipse = e;
      found = true;
    }
  }

  // Edge-based fallback within HSV mask region
  if (!found) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "No landmark found in contours, trying edge-based detection");
    cv::Mat gray;
    cv::cvtColor(work_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.2);
    cv::Mat edges;
    cv::Canny(gray, edges, 60, 180, 3, true);
    cv::bitwise_and(edges, mask, edges);
    cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(edges, edges, kd);

    std::vector<std::vector<cv::Point>> ctr2;
    cv::findContours(edges, ctr2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

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
  }

  if (!found) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "No landmark found after edge-based detection");
    return false;
  }

  // Recompute metrics for the selected ellipse (in work coordinates)
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

void CameraCalibrator::try_finalize_calibration() {
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

double CameraCalibrator::median(std::vector<double> v) {
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