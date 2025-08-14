// LineDetectorNode implementation

#include "src/line_detector_node.hpp"

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "src/camera_calibrator.hpp"

using std::placeholders::_1;

// Small helpers for angle conversion
static inline double deg2rad(double deg) { return deg * CV_PI / 180.0; }
static inline double rad2deg(double rad) { return rad * 180.0 / CV_PI; }

LineDetectorNode::LineDetectorNode() : Node("etrobo_line_detector") {
  RCLCPP_INFO(this->get_logger(), "Initializing LineDetectorNode");

  // Declare all parameters
  declare_all_parameters();

  // Initialize calibrator
  calibrator_ = std::make_unique<CameraCalibrator>(this);

  // Setup ROS entities
  setup_publishers();
  setup_subscription();
  setup_camera_info_subscription();
  setup_parameter_callback();

  RCLCPP_INFO(this->get_logger(), "LineDetectorNode initialized successfully");
}

void LineDetectorNode::declare_all_parameters() {
  // Topics
  image_topic_ =
      this->declare_parameter<std::string>("image_topic", "camera/image_raw");
  camera_info_topic_ = this->declare_parameter<std::string>(
      "camera_info_topic", "camera/camera_info");

  // Image preprocessing
  roi_ = this->declare_parameter<std::vector<int64_t>>("roi",
                                                       std::vector<int64_t>{});
  blur_ksize_ = this->declare_parameter<int>("blur_ksize", 5);
  blur_sigma_ = this->declare_parameter<double>("blur_sigma", 1.2);

  // Edge detection
  canny_low_ = this->declare_parameter<int>("canny_low", 50);
  canny_high_ = this->declare_parameter<int>("canny_high", 150);
  canny_aperture_ = this->declare_parameter<int>("canny_aperture", 3);
  canny_L2gradient_ = this->declare_parameter<bool>("canny_L2gradient", true);

  // Line detection
  rho_ = this->declare_parameter<double>("rho", 1.0);
  theta_deg_ = this->declare_parameter<double>("theta_deg", 1.0);
  threshold_ = this->declare_parameter<int>("threshold", 50);
  min_line_length_ = this->declare_parameter<double>("min_line_length", 100.0);
  max_line_gap_ = this->declare_parameter<double>("max_line_gap", 10.0);
  min_theta_deg_ = this->declare_parameter<double>("min_theta_deg", 0.0);
  max_theta_deg_ = this->declare_parameter<double>("max_theta_deg", 180.0);

  // HSV mask
  use_hsv_mask_ = this->declare_parameter<bool>("use_hsv_mask", false);
  hsv_lower_h_ = this->declare_parameter<int>("hsv_lower_h", 0);
  hsv_upper_h_ = this->declare_parameter<int>("hsv_upper_h", 180);
  hsv_lower_s_ = this->declare_parameter<int>("hsv_lower_s", 0);
  hsv_upper_s_ = this->declare_parameter<int>("hsv_upper_s", 255);
  hsv_lower_v_ = this->declare_parameter<int>("hsv_lower_v", 0);
  hsv_upper_v_ = this->declare_parameter<int>("hsv_upper_v", 100);
  hsv_dilate_kernel_ = this->declare_parameter<int>("hsv_dilate_kernel", 3);
  hsv_dilate_iter_ = this->declare_parameter<int>("hsv_dilate_iter", 1);

  // Visualization
  publish_image_ =
      this->declare_parameter<bool>("publish_image_with_lines", false);
  show_edges_ = this->declare_parameter<bool>("show_edges", false);
  draw_color_bgr_ = this->declare_parameter<std::vector<int64_t>>(
      "draw_color_bgr", {0, 255, 0});
  draw_thickness_ = this->declare_parameter<int>("draw_thickness", 2);

  // Localization parameters
  landmark_map_x_ = this->declare_parameter<double>("landmark_map_x", -0.409);
  landmark_map_y_ = this->declare_parameter<double>("landmark_map_y", 1.0);
}

void LineDetectorNode::sanitize_parameters() {
  // Clamp to valid ranges
  blur_ksize_ = std::max(1, blur_ksize_);
  if ((blur_ksize_ % 2) == 0) blur_ksize_++;  // Ensure odd kernel size
  blur_sigma_ = std::max(0.0, blur_sigma_);
  canny_low_ = std::max(0, canny_low_);
  canny_high_ = std::max(canny_low_, canny_high_);
  canny_aperture_ = std::max(3, canny_aperture_);
  if ((canny_aperture_ % 2) == 0) canny_aperture_++;
  if (canny_aperture_ > 7) canny_aperture_ = 7;

  rho_ = std::max(0.1, rho_);
  theta_deg_ = std::max(0.01, std::min(theta_deg_, 180.0));
  threshold_ = std::max(1, threshold_);
  min_line_length_ = std::max(1.0, min_line_length_);
  max_line_gap_ = std::max(0.0, max_line_gap_);

  if (min_theta_deg_ < 0.0) min_theta_deg_ = 0.0;
  if (max_theta_deg_ > 180.0) max_theta_deg_ = 180.0;
  if (min_theta_deg_ > max_theta_deg_) {
    std::swap(min_theta_deg_, max_theta_deg_);
  }

  // HSV ranges
  hsv_lower_h_ = std::max(0, std::min(179, hsv_lower_h_));
  hsv_upper_h_ = std::max(0, std::min(180, hsv_upper_h_));
  hsv_lower_s_ = std::max(0, std::min(255, hsv_lower_s_));
  hsv_upper_s_ = std::max(0, std::min(255, hsv_upper_s_));
  hsv_lower_v_ = std::max(0, std::min(255, hsv_lower_v_));
  hsv_upper_v_ = std::max(0, std::min(255, hsv_upper_v_));

  hsv_dilate_kernel_ = std::max(0, hsv_dilate_kernel_);
  hsv_dilate_iter_ = std::max(0, hsv_dilate_iter_);

  // Draw parameters
  draw_thickness_ = std::max(1, draw_thickness_);
  if (draw_color_bgr_.size() != 3) {
    draw_color_bgr_ = {0, 255, 0};  // Default green
  }
  for (auto& c : draw_color_bgr_) {
    c = std::max(int64_t(0), std::min(int64_t(255), c));
  }
}

void LineDetectorNode::setup_publishers() {
  // Use SensorData QoS but set RELIABLE to interoperate with image_view
  auto pub_qos = rclcpp::SensorDataQoS();
  pub_qos.reliable();
  if (publish_image_) {
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "image_with_lines", pub_qos);
  }
  lines_pub_ =
      this->create_publisher<std_msgs::msg::Float32MultiArray>("lines", 10);
}

void LineDetectorNode::setup_subscription() {
  // Subscription with SensorDataQoS depth=1, best effort
  auto qos = rclcpp::SensorDataQoS();
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, qos,
      std::bind(&LineDetectorNode::image_callback, this, _1));
}

void LineDetectorNode::setup_camera_info_subscription() {
  // CameraInfo is low rate; default QoS reliable
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::QoS(10),
      std::bind(&LineDetectorNode::camera_info_callback, this, _1));
}

void LineDetectorNode::setup_parameter_callback() {
  // Dynamic parameter updates for processing params (not topic/QoS)
  param_cb_handle_ = this->add_on_set_parameters_callback(std::bind(
      &LineDetectorNode::on_parameters_set, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult LineDetectorNode::on_parameters_set(
    const std::vector<rclcpp::Parameter>& parameters) {
  std::lock_guard<std::mutex> lock(param_mutex_);
  for (const auto& param : parameters) {
    // Try CameraCalibrator parameters first
    if (calibrator_ && calibrator_->try_update_parameter(param)) {
      continue;  // Parameter was handled by CameraCalibrator
    }

    const std::string& name = param.get_name();
    // Update processing parameters dynamically
    if (name == "blur_ksize")
      blur_ksize_ = param.as_int();
    else if (name == "blur_sigma")
      blur_sigma_ = param.as_double();
    else if (name == "canny_low")
      canny_low_ = param.as_int();
    else if (name == "canny_high")
      canny_high_ = param.as_int();
    else if (name == "canny_aperture")
      canny_aperture_ = param.as_int();
    else if (name == "canny_L2gradient")
      canny_L2gradient_ = param.as_bool();
    else if (name == "use_hsv_mask")
      use_hsv_mask_ = param.as_bool();
    else if (name == "show_edges")
      show_edges_ = param.as_bool();
    else if (name == "publish_image_with_lines") {
      publish_image_ = param.as_bool();
      if (publish_image_ && !image_pub_) {
        auto pub_qos = rclcpp::SensorDataQoS();
        pub_qos.reliable();
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "image_with_lines", pub_qos);
      }
    }
    // ... Add other parameters as needed
  }
  sanitize_parameters();
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}

void LineDetectorNode::camera_info_callback(
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

cv::Rect LineDetectorNode::valid_roi(const cv::Mat& img,
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

// ===== Main image processing callback =====
void LineDetectorNode::image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  const auto t0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(param_mutex_);

  // Step 1: Preprocess the image
  cv::Mat original_img, work_img;
  cv::Rect roi_rect;
  cv::Mat gray = preprocess_image(msg, original_img, work_img, roi_rect);
  if (gray.empty()) return;

  // Step 2: Detect gray dist
  cv::Point2d landmark_pos;
  bool found = false;
  if (state_ == State::Calibrating || state_ == State::Localizing) {
    found = calibrator_->process_frame(original_img, landmark_pos);
    if (state_ == State::Calibrating &&
        calibrator_->is_calibration_complete()) {
      // Calibration completed
      estimated_pitch_rad_ = calibrator_->get_estimated_pitch();
      state_ = State::Localizing;
      RCLCPP_INFO(this->get_logger(), "Calibration complete. Pitch: %.2f deg",
                  rad2deg(estimated_pitch_rad_));
    }
  }

  // Step 3: Edge detection
  cv::Mat edges = detect_edges(gray, work_img);

  // Step 4: Line detection
  std::vector<cv::Vec4i> segments_out;
  detect_lines(edges, segments_out);

  // Step 5: Publish lines data
  publish_lines(segments_out, roi_rect);

  // Step 6: Localization (if calibrated)
  if (state_ == State::Localizing) {
    perform_localization(segments_out, landmark_pos, found);
  }

  // Step 7: Visualization
  if (publish_image_ && image_pub_) {
    publish_visualization(msg, original_img, edges, segments_out, roi_rect);
  }

  // Step 8: Log timing
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 -
                                                                            t0)
          .count();

  if (state_ == State::Localizing && localization_valid_) {
    RCLCPP_INFO(this->get_logger(),
                "Robot pose: x=%.3f, y=%.3f, yaw=%.1f deg in %.2f ms",
                last_robot_x_, last_robot_y_, rad2deg(last_robot_yaw_), ms);
  } else if (state_ == State::Calibrating) {
    RCLCPP_DEBUG(this->get_logger(),
                 "Calibrating: %zu lines detected in %.2f ms",
                 segments_out.size(), ms);
  } else {
    RCLCPP_INFO(this->get_logger(), "Processed frame: %zu lines in %.2f ms",
                segments_out.size(), ms);
  }
}

cv::Mat LineDetectorNode::preprocess_image(
    const sensor_msgs::msg::Image::ConstSharedPtr msg, cv::Mat& original_img,
    cv::Mat& work_img, cv::Rect& roi_rect) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return cv::Mat();
  }

  original_img = cv_ptr->image;
  roi_rect = valid_roi(original_img, roi_);
  work_img = original_img(roi_rect).clone();

  cv::Mat gray;
  cv::cvtColor(work_img, gray, cv::COLOR_BGR2GRAY);

  if (blur_ksize_ > 1 && blur_sigma_ > 0) {
    cv::GaussianBlur(gray, gray, cv::Size(blur_ksize_, blur_ksize_),
                     blur_sigma_);
  }

  return gray;
}

cv::Mat LineDetectorNode::detect_edges(const cv::Mat& gray,
                                       const cv::Mat& work_img) {
  cv::Mat edges;
  cv::Canny(gray, edges, canny_low_, canny_high_, canny_aperture_,
            canny_L2gradient_);

  if (use_hsv_mask_) {
    cv::Mat hsv;
    cv::cvtColor(work_img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat hsv_mask;
    cv::inRange(hsv, cv::Scalar(hsv_lower_h_, hsv_lower_s_, hsv_lower_v_),
                cv::Scalar(hsv_upper_h_, hsv_upper_s_, hsv_upper_v_), hsv_mask);

    if (hsv_dilate_kernel_ > 0 && hsv_dilate_iter_ > 0) {
      cv::Mat kernel = cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(hsv_dilate_kernel_, hsv_dilate_kernel_));
      cv::dilate(hsv_mask, hsv_mask, kernel, cv::Point(-1, -1),
                 hsv_dilate_iter_);
    }

    cv::bitwise_and(edges, hsv_mask, edges);
  }

  return edges;
}

void LineDetectorNode::detect_lines(const cv::Mat& edges,
                                    std::vector<cv::Vec4i>& segments_out) {
  const double theta_rad = deg2rad(theta_deg_);
  const double min_theta_rad = deg2rad(min_theta_deg_);
  const double max_theta_rad = deg2rad(max_theta_deg_);

  if (min_theta_rad < max_theta_rad) {
    cv::HoughLinesP(edges, segments_out, rho_, theta_rad, threshold_,
                    min_line_length_, max_line_gap_);

    // Filter by angle
    segments_out.erase(
        std::remove_if(segments_out.begin(), segments_out.end(),
                       [min_theta_rad, max_theta_rad](const cv::Vec4i& line) {
                         int dx = line[2] - line[0];
                         int dy = line[3] - line[1];
                         double angle = std::atan2(std::abs(dy), std::abs(dx));
                         return (angle < min_theta_rad) ||
                                (angle > max_theta_rad);
                       }),
        segments_out.end());
  } else {
    cv::HoughLinesP(edges, segments_out, rho_, theta_rad, threshold_,
                    min_line_length_, max_line_gap_);
  }
}

void LineDetectorNode::publish_lines(const std::vector<cv::Vec4i>& segments_out,
                                     const cv::Rect& roi_rect) {
  auto msg = std_msgs::msg::Float32MultiArray();
  msg.layout.dim.resize(2);
  msg.layout.dim[0].label = "lines";
  msg.layout.dim[0].size = segments_out.size();
  msg.layout.dim[0].stride = segments_out.size() * 4;
  msg.layout.dim[1].label = "coords";
  msg.layout.dim[1].size = 4;
  msg.layout.dim[1].stride = 4;

  for (const auto& line : segments_out) {
    // Convert to original image coordinates
    msg.data.push_back(static_cast<float>(line[0] + roi_rect.x));
    msg.data.push_back(static_cast<float>(line[1] + roi_rect.y));
    msg.data.push_back(static_cast<float>(line[2] + roi_rect.x));
    msg.data.push_back(static_cast<float>(line[3] + roi_rect.y));
  }

  lines_pub_->publish(msg);
}

void LineDetectorNode::perform_localization(
    const std::vector<cv::Vec4i>& segments_out, const cv::Point2d& landmark_pos,
    bool found) {
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

  // Estimate yaw from detected lines
  double yaw_rad = 0.0;
  if (!segments_out.empty()) {
    std::vector<double> angles;
    for (const auto& seg : segments_out) {
      double dx = seg[2] - seg[0];
      double dy = seg[3] - seg[1];
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

void LineDetectorNode::publish_visualization(
    const sensor_msgs::msg::Image::ConstSharedPtr msg,
    const cv::Mat& original_img, const cv::Mat& edges,
    const std::vector<cv::Vec4i>& segments_out, const cv::Rect& roi_rect) {
  cv::Mat output_img;
  if (show_edges_) {
    cv::cvtColor(edges, output_img, cv::COLOR_GRAY2BGR);
  } else {
    output_img = original_img.clone();
  }

  // Draw ROI rectangle
  cv::rectangle(output_img, roi_rect, cv::Scalar(255, 255, 0), 1);

  // Draw detected lines
  const cv::Scalar line_color(draw_color_bgr_[0], draw_color_bgr_[1],
                              draw_color_bgr_[2]);
  for (const auto& line : segments_out) {
    cv::Point pt1(line[0] + roi_rect.x, line[1] + roi_rect.y);
    cv::Point pt2(line[2] + roi_rect.x, line[3] + roi_rect.y);
    cv::line(output_img, pt1, pt2, line_color, draw_thickness_);
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
