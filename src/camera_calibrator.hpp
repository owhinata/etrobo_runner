#ifndef ETROBO_LINE_DETECTOR__CAMERA_CALIBRATOR_HPP_
#define ETROBO_LINE_DETECTOR__CAMERA_CALIBRATOR_HPP_

#include <memory>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

// Forward declaration
class LineDetectorNode;

class CameraCalibrator {
 public:
  explicit CameraCalibrator(LineDetectorNode* node);
  ~CameraCalibrator();

  // Initialize and declare calibration parameters
  void declare_parameters();

  // Try to update a parameter if it belongs to CameraCalibrator
  bool try_update_parameter(const rclcpp::Parameter& param);

  // Set the current frame to process
  void set_process_frame(const cv::Mat& img);

  // Main processing method for calibration
  bool process_frame();

  // State queries
  bool is_calibration_complete() const;
  double get_estimated_pitch() const;

  // Get calibration parameters
  double get_camera_height() const;

  // Draw visualization overlay for image_with_lines output
  void draw_visualization_overlay(cv::Mat& img) const;

  // For localization: detect landmark in current frame
  bool detect_landmark_in_frame(double& x_full_out, double& v_full_out);

 private:
  // Forward declaration of implementation class
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif  // ETROBO_LINE_DETECTOR__CAMERA_CALIBRATOR_HPP_