// LineDetectorNode implementation with pimpl pattern

#include "src/line_detector_node.hpp"

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>

#include "src/adaptive_line_tracker.hpp"
#include "src/camera_calibrator.hpp"

using std::placeholders::_1;

// Small helpers for angle conversion
static inline double deg2rad(double deg) { return deg * CV_PI / 180.0; }
static inline double rad2deg(double rad) { return rad * 180.0 / CV_PI; }

// Helper function for ROI validation
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

// Implementation class
class LineDetectorNode::Impl {
 public:
  explicit Impl(LineDetectorNode* node);
  ~Impl() = default;

  void declare_all_parameters();
  void sanitize_parameters();
  void setup_publishers();
  void setup_subscription();
  void setup_camera_info_subscription();
  void setup_parameter_callback();

  rcl_interfaces::msg::SetParametersResult on_parameters_set(
      const std::vector<rclcpp::Parameter>& parameters);

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  LineDetectorNode::CameraIntrinsics get_camera_intrinsics() const;

 private:
  cv::Mat extract_black_regions(const cv::Mat& img);
  void configure_line_tracker();  // Helper to configure tracker parameters
  void publish_lines(
      const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines);
  void perform_localization(
      const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines,
      const cv::Point2d& landmark_pos, bool found);
  void publish_visualization(
      const sensor_msgs::msg::Image::ConstSharedPtr msg,
      const cv::Mat& original_img, const cv::Mat& edges,
      const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines,
      const cv::Rect& roi_rect,
      const std::vector<std::vector<cv::Point>>& contours);

  // Node reference
  LineDetectorNode* node_;

  // Subscriptions and publishers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lines_pub_;

  // Parameter callback
  rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // Topics
  std::string image_topic_;
  std::string camera_info_topic_;

  // Image preprocessing
  std::vector<int64_t> roi_;

  // HSV mask (only parameters actually used for black line detection)
  int hsv_upper_v_;  // Upper threshold for V channel to detect black
  int hsv_dilate_kernel_;
  int hsv_dilate_iter_;

  // Visualization
  bool publish_image_;
  bool show_edges_;
  bool show_contours_;

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

  // Adaptive line tracker instance
  std::unique_ptr<AdaptiveLineTracker> line_tracker_;

  // Localization state
  bool localization_valid_{false};
  double last_robot_x_{0.0};
  double last_robot_y_{0.0};
  double last_robot_yaw_{0.0};

  // Thread safety
  std::mutex param_mutex_;
};

// LineDetectorNode
LineDetectorNode::LineDetectorNode()
    : Node("etrobo_line_detector"), pimpl(std::make_unique<Impl>(this)) {
  RCLCPP_INFO(this->get_logger(), "LineDetectorNode initialized successfully");
}

LineDetectorNode::~LineDetectorNode() = default;

LineDetectorNode::CameraIntrinsics LineDetectorNode::get_camera_intrinsics()
    const {
  return pimpl->get_camera_intrinsics();
}

LineDetectorNode::Impl::Impl(LineDetectorNode* node) : node_(node) {
  RCLCPP_INFO(node_->get_logger(), "Initializing LineDetectorNode");

  // Declare all parameters
  declare_all_parameters();

  // Initialize calibrator
  calibrator_ = std::make_unique<CameraCalibrator>(node_);

  // Initialize adaptive line tracker with node pointer for logging
  line_tracker_ = std::make_unique<AdaptiveLineTracker>(node_);

  // Setup ROS entities
  setup_publishers();
  setup_subscription();
  setup_camera_info_subscription();
  setup_parameter_callback();
}

void LineDetectorNode::Impl::declare_all_parameters() {
  // Topics
  image_topic_ =
      node_->declare_parameter<std::string>("image_topic", "camera/image_raw");
  camera_info_topic_ = node_->declare_parameter<std::string>(
      "camera_info_topic", "camera/camera_info");

  // Image preprocessing
  roi_ = node_->declare_parameter<std::vector<int64_t>>("roi",
                                                        std::vector<int64_t>{});

  // HSV mask (only parameters actually used for black line detection)
  hsv_upper_v_ = node_->declare_parameter<int>("hsv_upper_v", 100);
  hsv_dilate_kernel_ = node_->declare_parameter<int>("hsv_dilate_kernel", 3);
  hsv_dilate_iter_ = node_->declare_parameter<int>("hsv_dilate_iter", 1);

  // Visualization
  publish_image_ =
      node_->declare_parameter<bool>("publish_image_with_lines", false);
  show_edges_ = node_->declare_parameter<bool>("show_edges", false);
  show_contours_ = node_->declare_parameter<bool>("show_contours", false);

  // Localization parameters
  landmark_map_x_ = node_->declare_parameter<double>("landmark_map_x", -0.409);
  landmark_map_y_ = node_->declare_parameter<double>("landmark_map_y", 1.0);
}

void LineDetectorNode::Impl::sanitize_parameters() {
  // HSV ranges (only parameters actually used)
  hsv_upper_v_ = std::max(0, std::min(255, hsv_upper_v_));
  hsv_dilate_kernel_ = std::max(0, hsv_dilate_kernel_);
  hsv_dilate_iter_ = std::max(0, hsv_dilate_iter_);
}

void LineDetectorNode::Impl::setup_publishers() {
  // Use SensorData QoS but set RELIABLE to interoperate with image_view
  auto pub_qos = rclcpp::SensorDataQoS();
  pub_qos.reliable();
  if (publish_image_) {
    image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        "image_with_lines", pub_qos);
  }
  lines_pub_ =
      node_->create_publisher<std_msgs::msg::Float32MultiArray>("lines", 10);
}

void LineDetectorNode::Impl::setup_subscription() {
  // Subscription with SensorDataQoS depth=1, best effort
  auto qos = rclcpp::SensorDataQoS();
  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, qos,
      std::bind(&LineDetectorNode::Impl::image_callback, this, _1));
}

void LineDetectorNode::Impl::setup_camera_info_subscription() {
  // CameraInfo is low rate; default QoS reliable
  camera_info_sub_ = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::QoS(10),
      std::bind(&LineDetectorNode::Impl::camera_info_callback, this, _1));
}

void LineDetectorNode::Impl::setup_parameter_callback() {
  // Dynamic parameter updates for processing params (not topic/QoS)
  param_cb_handle_ = node_->add_on_set_parameters_callback(std::bind(
      &LineDetectorNode::Impl::on_parameters_set, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult
LineDetectorNode::Impl::on_parameters_set(
    const std::vector<rclcpp::Parameter>& parameters) {
  std::lock_guard<std::mutex> lock(param_mutex_);
  for (const auto& param : parameters) {
    // Try CameraCalibrator parameters first
    if (calibrator_ && calibrator_->try_update_parameter(param)) {
      continue;  // Parameter was handled by CameraCalibrator
    }

    const std::string& name = param.get_name();
    // Update processing parameters dynamically
    if (name == "hsv_upper_v")
      hsv_upper_v_ = param.as_int();
    else if (name == "hsv_dilate_kernel")
      hsv_dilate_kernel_ = param.as_int();
    else if (name == "hsv_dilate_iter")
      hsv_dilate_iter_ = param.as_int();
    else if (name == "show_edges")
      show_edges_ = param.as_bool();
    else if (name == "show_contours")
      show_contours_ = param.as_bool();
    else if (name == "publish_image_with_lines") {
      publish_image_ = param.as_bool();
      if (publish_image_ && !image_pub_) {
        auto pub_qos = rclcpp::SensorDataQoS();
        pub_qos.reliable();
        image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
            "image_with_lines", pub_qos);
      }
    } else if (name == "roi")
      roi_ = param.as_integer_array();
    else if (name == "landmark_map_x")
      landmark_map_x_ = param.as_double();
    else if (name == "landmark_map_y")
      landmark_map_y_ = param.as_double();
  }
  sanitize_parameters();
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}

void LineDetectorNode::Impl::camera_info_callback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (msg->k.size() >= 9) {
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    has_cam_info_ = true;
  } else if (msg->p.size() >= 12) {
    fx_ = msg->p[0];
    fy_ = msg->p[5];
    cx_ = msg->p[2];
    cy_ = msg->p[6];
    has_cam_info_ = true;
  }
}

LineDetectorNode::CameraIntrinsics
LineDetectorNode::Impl::get_camera_intrinsics() const {
  return LineDetectorNode::CameraIntrinsics{has_cam_info_, fx_, fy_, cx_, cy_};
}

void LineDetectorNode::Impl::image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  const auto t0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(param_mutex_);

  // Step 1: Convert ROS image to OpenCV format
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat original_img = cv_ptr->image;
  cv::Rect roi_rect = valid_roi(original_img, roi_);
  cv::Mat work_img = original_img(roi_rect).clone();

  // Step 2: Detect gray disk
  cv::Point2d landmark_pos;
  bool found = false;
  if (state_ == State::Calibrating || state_ == State::Localizing) {
    found = calibrator_->process_frame(original_img, landmark_pos);
    if (state_ == State::Calibrating &&
        calibrator_->is_calibration_complete()) {
      // Calibration completed
      estimated_pitch_rad_ = calibrator_->get_estimated_pitch();
      state_ = State::Localizing;
      RCLCPP_INFO(node_->get_logger(), "Calibration complete. Pitch: %.2f deg",
                  rad2deg(estimated_pitch_rad_));
    }
  }

  // Step 3: Line detection using adaptive tracker
  // Extract black regions only from ROI area (work_img is already ROI-cropped)
  cv::Mat black_mask = extract_black_regions(work_img);

  // Find contours for visualization
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat mask_for_contours = black_mask.clone();
  cv::findContours(mask_for_contours, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // Configure and track the black line (gray disk is already excluded by HSV
  // thresholding)
  // Calculate current processing time to pass to tracker
  const auto t_current = std::chrono::steady_clock::now();
  const double current_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          t_current - t0)
          .count();

  configure_line_tracker();
  auto tracked_lines_rel = line_tracker_->track_line(black_mask, current_ms);

  // Convert from relative (mask) coordinates to absolute (image) coordinates
  std::vector<AdaptiveLineTracker::TrackedLine> tracked_lines;
  for (const auto& line_rel : tracked_lines_rel) {
    AdaptiveLineTracker::TrackedLine line_abs;
    line_abs.contour_id = line_rel.contour_id;
    line_abs.area = line_rel.area;
    line_abs.points.reserve(line_rel.points.size());
    for (const auto& pt : line_rel.points) {
      line_abs.points.emplace_back(pt.x + roi_rect.x, pt.y + roi_rect.y);
    }
    tracked_lines.push_back(line_abs);
  }

  // Frame counter for periodic logging (handled in tracker)

  // Step 4: Publish lines data
  publish_lines(tracked_lines);

  // Step 5: Localization (if calibrated)
  if (state_ == State::Localizing) {
    perform_localization(tracked_lines, landmark_pos, found);
  }

  // Step 6: Visualization
  if (publish_image_ && image_pub_) {
    publish_visualization(msg, original_img, black_mask, tracked_lines,
                          roi_rect, contours);
  }

  // Step 8: Log timing
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 -
                                                                            t0)
          .count();

  if (state_ == State::Localizing && localization_valid_) {
    RCLCPP_INFO(node_->get_logger(),
                "Robot pose: x=%.3f, y=%.3f, yaw=%.1f deg in %.2f ms",
                last_robot_x_, last_robot_y_, rad2deg(last_robot_yaw_), ms);
  } else if (state_ == State::Calibrating) {
    // Count total points for logging
    size_t total_points = 0;
    for (const auto& line : tracked_lines) {
      total_points += line.points.size();
    }
    RCLCPP_DEBUG(node_->get_logger(),
                 "Calibrating: %zu points tracked in %.2f ms", total_points,
                 ms);
  }
}

cv::Mat LineDetectorNode::Impl::extract_black_regions(const cv::Mat& img) {
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
  // Black line typically has V < 50-80 depending on lighting

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

void LineDetectorNode::Impl::configure_line_tracker() {
  if (!line_tracker_) {
    RCLCPP_WARN(node_->get_logger(), "Line tracker not initialized");
    return;
  }

  // Configure tracker (can be made dynamic via parameters in the future)
  AdaptiveLineTracker::Config config;
  config.max_line_width =
      50.0;  // Strict: black line should be 20-40 pixels typically
  config.min_line_width = 6.0;     // Allow thinner lines to be detected
  config.max_lateral_jump = 20.0;  // Reduced: line shouldn't jump too much
  config.scan_step = 5;
  config.position_weight = 0.5;  // Increase weight for position continuity
  config.prediction_weight = 0.3;
  config.width_weight = 0.2;
  line_tracker_->set_config(config);
}

void LineDetectorNode::Impl::publish_lines(
    const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines) {
  auto msg = std_msgs::msg::Float32MultiArray();

  // Collect all points from all lines for backward compatibility
  std::vector<cv::Point2d> all_points;
  for (const auto& line : tracked_lines) {
    for (const auto& pt : line.points) {
      all_points.push_back(pt);
    }
  }

  // Convert points to segments for backward compatibility
  if (all_points.size() >= 2) {
    size_t num_segments = all_points.size() - 1;
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "lines";
    msg.layout.dim[0].size = num_segments;
    msg.layout.dim[0].stride = num_segments * 4;
    msg.layout.dim[1].label = "coords";
    msg.layout.dim[1].size = 4;
    msg.layout.dim[1].stride = 4;

    for (size_t i = 1; i < all_points.size(); i++) {
      // Points are already in absolute coordinates
      msg.data.push_back(static_cast<float>(all_points[i - 1].x));
      msg.data.push_back(static_cast<float>(all_points[i - 1].y));
      msg.data.push_back(static_cast<float>(all_points[i].x));
      msg.data.push_back(static_cast<float>(all_points[i].y));
    }
  }

  lines_pub_->publish(msg);
}

void LineDetectorNode::Impl::perform_localization(
    const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines,
    const cv::Point2d& landmark_pos, bool found) {
  if (!found || !has_cam_info_ || std::isnan(estimated_pitch_rad_)) {
    localization_valid_ = false;
    return;
  }

  // Compute robot position from landmark
  const double u = (landmark_pos.y - cy_) / fy_;
  const double phi = estimated_pitch_rad_;
  const double tan_phi = std::tan(phi);
  const double h = calibrator_->get_camera_height();

  // Distance to landmark
  const double denom = (tan_phi + u);
  if (std::abs(denom) < 1e-6) {
    localization_valid_ = false;
    return;
  }
  const double d = h / denom;

  // Lateral offset from image x coordinate
  const double x_offset = (landmark_pos.x - cx_) / fx_ * d;

  // Robot position relative to landmark
  const double robot_x = landmark_map_x_ - x_offset;
  const double robot_y = landmark_map_y_ - d;

  // Estimate yaw from tracked points trajectory
  // Collect all points from all lines
  std::vector<cv::Point2d> all_points;
  for (const auto& line : tracked_lines) {
    for (const auto& pt : line.points) {
      all_points.push_back(pt);
    }
  }

  double yaw_rad = 0.0;
  if (all_points.size() >= 2) {
    std::vector<double> angles;
    for (size_t i = 1; i < all_points.size(); i++) {
      double dx = all_points[i].x - all_points[i - 1].x;
      double dy = all_points[i].y - all_points[i - 1].y;
      if (std::abs(dx) > 10) {  // Filter short segments
        angles.push_back(std::atan2(dy, dx));
      }
    }

    if (!angles.empty()) {
      // Use median angle
      std::nth_element(angles.begin(), angles.begin() + angles.size() / 2,
                       angles.end());
      yaw_rad = angles[angles.size() / 2];
    }
  }

  last_robot_x_ = robot_x;
  last_robot_y_ = robot_y;
  last_robot_yaw_ = yaw_rad;
  localization_valid_ = true;
}

void LineDetectorNode::Impl::publish_visualization(
    const sensor_msgs::msg::Image::ConstSharedPtr msg,
    const cv::Mat& original_img, const cv::Mat& edges,
    const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines,
    const cv::Rect& roi_rect,
    const std::vector<std::vector<cv::Point>>& contours) {
  cv::Mat output_img;
  if (show_edges_) {
    cv::cvtColor(edges, output_img, cv::COLOR_GRAY2BGR);
  } else {
    output_img = original_img.clone();
  }

  // Draw ROI rectangle
  cv::rectangle(output_img, roi_rect, cv::Scalar(255, 255, 0), 1);

  // Draw contours only if show_contours is enabled
  if (show_contours_) {
    // Draw only valid contours (area >= 20)
    const double MIN_CONTOUR_AREA = 20.0;  // Same threshold as in tracker
    int valid_contour_idx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
      // Skip small contours
      double area = cv::contourArea(contours[i]);
      if (area < MIN_CONTOUR_AREA) {
        continue;  // Don't visualize small contours
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

      // Draw contour outline (thicker line for better visibility)
      cv::drawContours(output_img, contour_to_draw, 0, contour_color, 2);
      valid_contour_idx++;
    }
  }

  // Draw tracked points with different colors for different contours
  for (size_t line_idx = 0; line_idx < tracked_lines.size(); line_idx++) {
    const auto& line = tracked_lines[line_idx];

    // Use different colors for different contours
    cv::Scalar point_color;
    switch (line.contour_id % 6) {
      case 0:
        point_color = cv::Scalar(0, 255, 0);  // Green
        break;
      case 1:
        point_color = cv::Scalar(0, 165, 255);  // Orange
        break;
      case 2:
        point_color = cv::Scalar(255, 255, 0);  // Cyan
        break;
      case 3:
        point_color = cv::Scalar(255, 0, 255);  // Magenta
        break;
      case 4:
        point_color = cv::Scalar(0, 255, 255);  // Yellow
        break;
      case 5:
        point_color = cv::Scalar(255, 128, 128);  // Light Blue
        break;
    }

    for (const auto& pt : line.points) {
      // Points are already in absolute coordinates
      cv::Point center(static_cast<int>(pt.x), static_cast<int>(pt.y));
      cv::circle(output_img, center, 3, point_color, -1);  // Filled circle
    }
  }

  // Draw calibration visualization
  if (state_ == State::Calibrating) {
    calibrator_->draw_visualization_overlay(output_img);
  }

  // Add localization info
  if (state_ == State::Localizing && localization_valid_) {
    cv::putText(output_img,
                cv::format("Pose: x=%.2f y=%.2f yaw=%.1f", last_robot_x_,
                           last_robot_y_, rad2deg(last_robot_yaw_)),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
  }

  // Publish the image
  cv_bridge::CvImage out_msg;
  out_msg.header = msg->header;
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = output_img;
  image_pub_->publish(*out_msg.toImageMsg());
}
