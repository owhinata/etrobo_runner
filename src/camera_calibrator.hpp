#ifndef ETROBO_LINE_DETECTOR__CAMERA_CALIBRATOR_HPP_
#define ETROBO_LINE_DETECTOR__CAMERA_CALIBRATOR_HPP_

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>

// Forward declaration
class LineDetectorNode;

class CameraCalibrator {
 public:
  explicit CameraCalibrator(LineDetectorNode* node);

  // Initialize and declare calibration parameters
  void declare_parameters();

  // Try to update a parameter if it belongs to CameraCalibrator
  bool try_update_parameter(const rclcpp::Parameter& param);

  // Set the current frame to process
  void set_process_frame(const cv::Mat& img);

  // Main processing method for calibration
  bool process_frame();

  // State queries
  bool is_calibration_complete() const { return calibration_complete_; }
  double get_estimated_pitch() const { return estimated_pitch_rad_; }

  // Get calibration parameters
  double get_camera_height() const { return camera_height_m_; }

  // Draw visualization overlay for image_with_lines output
  void draw_visualization_overlay(cv::Mat& img) const;

  // For localization: detect landmark in current frame
  bool detect_landmark_in_frame(double& x_full_out, double& v_full_out);

 private:
  // Internal utility methods
  bool detect_landmark_center(double& x_full_out, double& v_full_out);
  cv::Mat create_hsv_mask(const cv::Mat& bgr_img);
  bool find_ellipse_from_contours(
      const std::vector<std::vector<cv::Point>>& contours,
      const cv::Mat& bgr_img, const cv::Mat& hsv_img, const cv::Mat& mask,
      cv::RotatedRect& best_ellipse);
  bool find_ellipse_edge_based(const cv::Mat& bgr_img, const cv::Mat& mask,
                               cv::RotatedRect& best_ellipse);
  double score_ellipse(const cv::RotatedRect& ellipse, const cv::Mat& bgr_img,
                       const cv::Mat& hsv_img, const cv::Mat&);
  cv::Rect valid_roi(const cv::Mat& img, const std::vector<int64_t>& roi);
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

#endif  // ETROBO_LINE_DETECTOR__CAMERA_CALIBRATOR_HPP_