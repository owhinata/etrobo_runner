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
  void publish_lines(
      const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines);
  void perform_localization(
      const std::vector<AdaptiveLineTracker::TrackedLine>& tracked_lines,
      const cv::Point2d& landmark_pos, bool found);
  void publish_visualization(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                             const cv::Mat& original_img);

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

  // Visualization
  bool publish_image_;

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

  // Initialize components first
  calibrator_ = std::make_unique<CameraCalibrator>(node_);
  line_tracker_ = std::make_unique<AdaptiveLineTracker>(node_);

  // Declare all parameters (including components)
  declare_all_parameters();

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

  // Visualization
  publish_image_ =
      node_->declare_parameter<bool>("publish_image_with_lines", false);

  // Localization parameters
  landmark_map_x_ = node_->declare_parameter<double>("landmark_map_x", -0.409);
  landmark_map_y_ = node_->declare_parameter<double>("landmark_map_y", 1.0);

  // Declare parameters for components
  if (calibrator_) {
    calibrator_->declare_parameters();
  }
  if (line_tracker_) {
    line_tracker_->declare_parameters();
  }
}

void LineDetectorNode::Impl::sanitize_parameters() {
  // Currently no parameters need sanitization in LineDetectorNode
  // HSV parameters are handled by AdaptiveLineTracker
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

    // Try AdaptiveLineTracker parameters
    if (line_tracker_ && line_tracker_->try_update_parameter(param)) {
      continue;  // Parameter was handled by AdaptiveLineTracker
    }

    const std::string& name = param.get_name();
    // Update processing parameters dynamically
    if (name == "publish_image_with_lines") {
      publish_image_ = param.as_bool();
      if (publish_image_ && !image_pub_) {
        auto pub_qos = rclcpp::SensorDataQoS();
        pub_qos.reliable();
        image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
            "image_with_lines", pub_qos);
      }
    } else if (name == "landmark_map_x")
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
  AdaptiveLineTracker::DetectionResult detection_result;
  line_tracker_->process_frame(original_img, detection_result);

  // Convert tracked lines to absolute coordinates (already in absolute
  // coordinates)
  std::vector<AdaptiveLineTracker::TrackedLine> tracked_lines =
      detection_result.tracked_lines;

  // Step 4: Log detection results
  const auto t1 = std::chrono::steady_clock::now();
  const double total_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 -
                                                                            t0)
          .count();

  // Build points string
  std::string points_info = " [points:";
  for (size_t i = 0; i < detection_result.segment_counts.size(); i++) {
    if (i > 0) points_info += ",";
    points_info += std::to_string(detection_result.segment_counts[i]);
  }
  points_info += "]";

  RCLCPP_INFO(node_->get_logger(),
              "Detection: %d/%d scans, %d/%d contours valid%s in %.2f ms",
              detection_result.successful_detections,
              detection_result.total_scans, detection_result.valid_contours,
              detection_result.total_contours, points_info.c_str(), total_ms);

  // Step 5: Publish lines data
  publish_lines(tracked_lines);

  // Step 6: Localization (if calibrated)
  if (state_ == State::Localizing) {
    perform_localization(tracked_lines, landmark_pos, found);
  }

  // Step 7: Visualization
  if (publish_image_ && image_pub_) {
    publish_visualization(msg, original_img);
  }

  // Step 8: Additional logging for specific states
  if (state_ == State::Localizing && localization_valid_) {
    RCLCPP_INFO(node_->get_logger(),
                "Robot pose: x=%.3f, y=%.3f, yaw=%.1f deg in %.2f ms",
                last_robot_x_, last_robot_y_, rad2deg(last_robot_yaw_),
                total_ms);
  } else if (state_ == State::Calibrating) {
    // Count total points for logging
    size_t total_points = 0;
    for (const auto& line : tracked_lines) {
      total_points += line.points.size();
    }
    RCLCPP_DEBUG(node_->get_logger(),
                 "Calibrating: %zu points tracked in %.2f ms", total_points,
                 total_ms);
  }
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
    const cv::Mat& original_img) {
  cv::Mat output_img = original_img.clone();

  // Delegate visualization to AdaptiveLineTracker
  line_tracker_->draw_visualization_overlay(output_img);

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
