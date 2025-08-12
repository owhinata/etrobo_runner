#ifndef ETROBO_LINE_DETECTOR__LINE_DETECTOR_NODE_HPP_
#define ETROBO_LINE_DETECTOR__LINE_DETECTOR_NODE_HPP_

#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>

#include "src/camera_calibrator.hpp"

class LineDetectorNode : public rclcpp::Node {
 public:
  enum class State { Calibrating, Localizing };

  LineDetectorNode();

  // Friend class for CameraCalibrator to access private members
  friend class CameraCalibrator;

 private:
  void declare_all_parameters();
  void sanitize_parameters();
  void setup_publishers();
  void setup_subscription();
  void setup_camera_info_subscription();
  void setup_parameter_callback();

  rcl_interfaces::msg::SetParametersResult on_parameters_set(
      const std::vector<rclcpp::Parameter>& parameters);

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  cv::Rect valid_roi(const cv::Mat& img, const std::vector<int64_t>& roi);

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  cv::Mat preprocess_image(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                           cv::Mat& original_img, cv::Mat& work_img,
                           cv::Rect& roi_rect);

  cv::Mat detect_edges(const cv::Mat& gray, const cv::Mat& work_img);

  void detect_lines(const cv::Mat& edges, std::vector<cv::Vec4i>& segments_out);

  void publish_lines(const std::vector<cv::Vec4i>& segments_out,
                     const cv::Rect& roi_rect);

  void perform_localization(const std::vector<cv::Vec4i>& segments_out,
                            const cv::Mat& original_img);

  void publish_visualization(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                             const cv::Mat& original_img,
                             const cv::Mat& work_img, const cv::Mat& edges,
                             const std::vector<cv::Vec4i>& segments_out,
                             const cv::Rect& roi_rect);

  // Subscriptions and publishers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lines_pub_;

  // Parameter callback
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // Topics
  std::string image_topic_;
  std::string camera_info_topic_;

  // Image preprocessing
  std::vector<int64_t> roi_;
  int blur_ksize_;
  double blur_sigma_;

  // Edge detection
  int canny_low_;
  int canny_high_;
  int canny_aperture_;
  bool canny_L2gradient_;

  // Line detection
  double rho_;
  double theta_deg_;
  int threshold_;
  double min_line_length_;
  double max_line_gap_;
  double min_theta_deg_;
  double max_theta_deg_;

  // HSV mask
  bool use_hsv_mask_;
  int hsv_lower_h_;
  int hsv_upper_h_;
  int hsv_lower_s_;
  int hsv_upper_s_;
  int hsv_lower_v_;
  int hsv_upper_v_;
  int hsv_dilate_kernel_;
  int hsv_dilate_iter_;

  // Visualization
  bool publish_image_;
  bool show_edges_;
  std::vector<int64_t> draw_color_bgr_;
  int draw_thickness_;

  // Localization parameters
  double landmark_map_x_;
  double landmark_map_y_;

  // Camera intrinsics
  bool has_cam_info_{false};
  double fx_{1.0}, fy_{1.0}, cx_{0.0}, cy_{0.0};

  // Processing state
  State state_{State::Calibrating};
  double estimated_pitch_rad_{std::numeric_limits<double>::quiet_NaN()};

  // Calibrator instance
  std::unique_ptr<CameraCalibrator> calibrator_;

  // Localization state
  bool localization_valid_{false};
  double last_robot_x_{0.0};
  double last_robot_y_{0.0};
  double last_robot_yaw_{0.0};

  // Thread safety
  std::mutex param_mutex_;
};

#endif  // ETROBO_LINE_DETECTOR__LINE_DETECTOR_NODE_HPP_