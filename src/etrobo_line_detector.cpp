// Minimal single-file implementation per doc/DESIGN.md

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>

using std::placeholders::_1;

// Small helpers for angle conversion
static inline double deg2rad(double deg) { return deg * CV_PI / 180.0; }
static inline double rad2deg(double rad) { return rad * 180.0 / CV_PI; }

// Forward declaration
class LineDetectorNode;

class CameraCalibrator {
 public:
  CameraCalibrator(LineDetectorNode* node);

  // Main processing method
  bool process_frame(const cv::Mat& img);

  // State queries
  bool is_calibration_complete() const { return calibration_complete_; }
  double get_estimated_pitch() const { return estimated_pitch_rad_; }

  // Visualization data access
  bool has_valid_circle() const { return last_circle_valid_; }
  bool has_valid_ellipse() const { return last_ellipse_valid_; }
  cv::Point2d get_last_circle() const { return last_circle_px_; }
  cv::RotatedRect get_last_ellipse() const { return last_ellipse_full_; }
  cv::Mat get_last_hsv_mask() const { return last_calib_hsv_mask_; }

  // Metrics for visualization
  double get_last_angle_deg() const { return last_angle_deg_; }
  double get_last_ratio() const { return last_ratio_; }
  double get_last_mean_s() const { return last_mean_s_; }
  double get_last_mean_v() const { return last_mean_v_; }

  bool detect_landmark_center(const cv::Mat& work_img, const cv::Rect& roi,
                              double scale, double& x_full_out,
                              double& v_full_out);

 private:
  // Internal utility methods
  cv::Rect valid_roi(const cv::Mat& img, const std::vector<int64_t>& roi);
  void try_finalize_calibration();
  static double median(std::vector<double> v);

  LineDetectorNode* node_;

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

  // Parameter cache (fetched from node)
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

  // Internal methods moved to CameraCalibrator
};

class LineDetectorNode : public rclcpp::Node {
 public:
  enum class State { Calibrating, Localizing };

  LineDetectorNode() : Node("etrobo_line_detector") {
    declare_parameters();
    sanitize_parameters();
    setup_publishers();
    setup_subscription();
    setup_camera_info_subscription();
    setup_parameter_callback();

    // Initialize calibrator after parameters are set
    calibrator_ = std::make_unique<CameraCalibrator>(this);

    RCLCPP_INFO(this->get_logger(),
                "Line detector node initialized. Subscribing to: %s",
                image_topic_.c_str());
  }

  // Make CameraCalibrator a friend class to access private members
  friend class CameraCalibrator;

 private:
  // ===== Core methods =====
  void declare_parameters() {
    image_topic_ = this->declare_parameter<std::string>("image_topic", "image");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "camera_info");

    publish_image_ =
        this->declare_parameter<bool>("publish_image_with_lines", false);
    show_edges_ = this->declare_parameter<bool>("show_edges", false);

    use_hsv_mask_ = this->declare_parameter<bool>("use_hsv_mask", true);
    blur_ksize_ = this->declare_parameter<int>("blur_ksize", 5);
    blur_sigma_ = this->declare_parameter<double>("blur_sigma", 1.5);
    roi_ = this->declare_parameter<std::vector<int64_t>>(
        "roi", std::vector<int64_t>{-1, -1, -1, -1});

    canny_low_ = this->declare_parameter<int>("canny_low", 40);
    canny_high_ = this->declare_parameter<int>("canny_high", 120);
    canny_aperture_ = this->declare_parameter<int>("canny_aperture", 3);
    canny_L2gradient_ =
        this->declare_parameter<bool>("canny_L2gradient", false);

    hough_type_ = this->declare_parameter<std::string>(
        "hough_type", std::string("probabilistic"));
    rho_ = this->declare_parameter<double>("rho", 1.0);
    theta_deg_ = this->declare_parameter<double>("theta_deg", 1.0);
    threshold_ = this->declare_parameter<int>("threshold", 50);
    min_line_length_ = this->declare_parameter<double>("min_line_length", 30.0);
    max_line_gap_ = this->declare_parameter<double>("max_line_gap", 10.0);
    min_theta_deg_ = this->declare_parameter<double>("min_theta_deg", 0.0);
    max_theta_deg_ = this->declare_parameter<double>("max_theta_deg", 180.0);

    draw_color_bgr_ = this->declare_parameter<std::vector<int64_t>>(
        "draw_color_bgr", std::vector<int64_t>{0, 255, 0});
    draw_thickness_ = this->declare_parameter<int>("draw_thickness", 2);

    // HSV mask parameters
    hsv_lower_h_ = this->declare_parameter<int>("hsv_lower_h", 0);
    hsv_lower_s_ = this->declare_parameter<int>("hsv_lower_s", 0);
    hsv_lower_v_ = this->declare_parameter<int>("hsv_lower_v", 0);
    hsv_upper_h_ = this->declare_parameter<int>("hsv_upper_h", 180);
    hsv_upper_s_ = this->declare_parameter<int>("hsv_upper_s", 40);
    hsv_upper_v_ = this->declare_parameter<int>("hsv_upper_v", 148);
    hsv_dilate_kernel_ = this->declare_parameter<int>("hsv_dilate_kernel", 3);
    hsv_dilate_iter_ = this->declare_parameter<int>("hsv_dilate_iter", 2);

    // Calibration parameters
    camera_height_m_ =
        this->declare_parameter<double>("camera_height_meters", 0.2);
    landmark_distance_m_ =
        this->declare_parameter<double>("landmark_distance_meters", 0.59);
    calib_timeout_sec_ =
        this->declare_parameter<double>("calib_timeout_sec", 60.0);
    calib_roi_ = this->declare_parameter<std::vector<int64_t>>(
        "calib_roi", std::vector<int64_t>{200, 150, 240, 180});
    calib_hsv_s_max_ = this->declare_parameter<int>("calib_hsv_s_max", 16);
    calib_hsv_v_min_ = this->declare_parameter<int>("calib_hsv_v_min", 100);
    calib_hsv_v_max_ = this->declare_parameter<int>("calib_hsv_v_max", 168);
    calib_min_area_ = this->declare_parameter<int>("calib_min_area", 80);
    calib_min_major_px_ = this->declare_parameter<int>("calib_min_major_px", 8);
    calib_max_major_ratio_ =
        this->declare_parameter<double>("calib_max_major_ratio", 0.65);
    calib_fill_min_ = this->declare_parameter<double>("calib_fill_min", 0.25);

    // Localization parameters
    landmark_map_x_ = this->declare_parameter<double>("landmark_map_x", -0.409);
    landmark_map_y_ = this->declare_parameter<double>("landmark_map_y", 1.0);
  }

  void sanitize_parameters() {
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

    // Calibration parameters
    camera_height_m_ = std::max(0.01, camera_height_m_);
    landmark_distance_m_ = std::max(0.01, landmark_distance_m_);
    calib_timeout_sec_ = std::max(0.0, calib_timeout_sec_);

    calib_hsv_s_max_ = std::max(0, std::min(255, calib_hsv_s_max_));
    calib_hsv_v_min_ = std::max(0, std::min(255, calib_hsv_v_min_));
    calib_hsv_v_max_ =
        std::max(calib_hsv_v_min_, std::min(255, calib_hsv_v_max_));
    calib_min_area_ = std::max(1, calib_min_area_);
    calib_min_major_px_ = std::max(1, calib_min_major_px_);
    calib_max_major_ratio_ =
        std::max(0.1, std::min(1.0, calib_max_major_ratio_));
    calib_fill_min_ = std::max(0.0, std::min(1.0, calib_fill_min_));
  }

  void setup_publishers() {
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

  void setup_subscription() {
    // Subscription with SensorDataQoS depth=1, best effort
    auto qos = rclcpp::SensorDataQoS();
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, qos,
        std::bind(&LineDetectorNode::image_callback, this, _1));
  }

  void setup_camera_info_subscription() {
    // CameraInfo is low rate; default QoS reliable
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::QoS(10),
        std::bind(&LineDetectorNode::camera_info_callback, this, _1));
  }

  void setup_parameter_callback() {
    // Dynamic parameter updates for processing params (not topic/QoS)
    param_cb_handle_ = this->add_on_set_parameters_callback(std::bind(
        &LineDetectorNode::on_parameters_set, this, std::placeholders::_1));
  }

  rcl_interfaces::msg::SetParametersResult on_parameters_set(
      const std::vector<rclcpp::Parameter>& parameters) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    for (const auto& param : parameters) {
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

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
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

  cv::Rect valid_roi(const cv::Mat& img, const std::vector<int64_t>& roi) {
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
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(param_mutex_);

    // Step 1: Preprocess the image
    cv::Mat original_img, work_img;
    cv::Rect roi_rect;
    cv::Mat gray = preprocess_image(msg, original_img, work_img, roi_rect);
    if (gray.empty()) return;

    // Step 2: Calibration phase
    if (state_ == State::Calibrating) {
      RCLCPP_DEBUG(this->get_logger(), "Processing frame in calibration mode");
      if (calibrator_->process_frame(original_img)) {
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
      perform_localization(segments_out, original_img);
    }

    // Step 7: Visualization
    if (publish_image_ && image_pub_) {
      publish_visualization(msg, original_img, work_img, edges, segments_out,
                            roi_rect);
    }

    // Step 8: Log timing
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0)
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

  cv::Mat preprocess_image(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                           cv::Mat& original_img, cv::Mat& work_img,
                           cv::Rect& roi_rect) {
    // Convert to cv::Mat
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      if (msg->encoding == sensor_msgs::image_encodings::BGR8 ||
          msg->encoding == sensor_msgs::image_encodings::MONO8) {
        cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
      } else {
        // Convert all other encodings (including RGB8) to BGR8
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      }
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
      return cv::Mat();
    }

    cv::Mat img = cv_ptr->image;
    original_img = img;

    // ROI (no downscaling)
    roi_rect = valid_roi(img, roi_);
    cv::Mat work = img(roi_rect).clone();
    work_img = work;  // save work image

    // Convert to grayscale for Canny edge detection
    cv::Mat gray;
    if (work.channels() == 3) {
      cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = work.clone();  // clone to avoid sharing the same pointer
    }

    // Apply blur if enabled
    if (blur_ksize_ > 1) {
      cv::GaussianBlur(gray, gray, cv::Size(blur_ksize_, blur_ksize_),
                       blur_sigma_);
    }

    return gray;
  }

  cv::Mat detect_edges(const cv::Mat& gray, const cv::Mat& work) {
    // Canny
    cv::Mat edges;
    cv::Canny(gray, edges, canny_low_, canny_high_, canny_aperture_,
              canny_L2gradient_);

    // Optional HSV mask AFTER Canny to avoid weakening edges before gradient
    if (use_hsv_mask_) {
      cv::Mat hsv_src;
      if (work.channels() == 1) {
        cv::cvtColor(work, hsv_src, cv::COLOR_GRAY2BGR);
        cv::cvtColor(hsv_src, hsv_src, cv::COLOR_BGR2HSV);
      } else {
        cv::cvtColor(work, hsv_src, cv::COLOR_BGR2HSV);
      }
      const cv::Scalar lower(hsv_lower_h_, hsv_lower_s_, hsv_lower_v_);
      const cv::Scalar upper(hsv_upper_h_, hsv_upper_s_, hsv_upper_v_);
      cv::Mat hsv_mask;
      cv::inRange(hsv_src, lower, upper, hsv_mask);

      // Morphological operations
      if (hsv_dilate_kernel_ > 0 && hsv_dilate_iter_ > 0) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(hsv_dilate_kernel_, hsv_dilate_kernel_));
        cv::dilate(hsv_mask, hsv_mask, kernel, cv::Point(-1, -1),
                   hsv_dilate_iter_);
      }

      // Mask the edges
      cv::bitwise_and(edges, hsv_mask, edges);
    }

    return edges;
  }

  void detect_lines(const cv::Mat& edges,
                    std::vector<cv::Vec4i>& segments_out) {
    segments_out.clear();
    const double theta_rad = deg2rad(theta_deg_);
    const double min_theta_rad = deg2rad(min_theta_deg_);
    const double max_theta_rad = deg2rad(max_theta_deg_);

    if (hough_type_ == "standard") {
      // Standard Hough
      std::vector<cv::Vec2f> lines;
      cv::HoughLines(edges, lines, rho_, theta_rad, threshold_, 0, 0,
                     min_theta_rad, max_theta_rad);
      // Convert polar to Cartesian
      for (const auto& line : lines) {
        float r = line[0], t = line[1];
        double a = std::cos(t), b = std::sin(t);
        double x0 = a * r, y0 = b * r;
        cv::Vec4i seg;
        seg[0] = cvRound(x0 + 1000 * (-b));
        seg[1] = cvRound(y0 + 1000 * (a));
        seg[2] = cvRound(x0 - 1000 * (-b));
        seg[3] = cvRound(y0 - 1000 * (a));
        segments_out.push_back(seg);
      }
    } else {
      // Probabilistic Hough (default)
      cv::HoughLinesP(edges, segments_out, rho_, theta_rad, threshold_,
                      min_line_length_, max_line_gap_);
      // Filter by angle
      if (min_theta_deg_ > 0.0 || max_theta_deg_ < 180.0) {
        std::vector<cv::Vec4i> filtered;
        for (const auto& seg : segments_out) {
          double dx = seg[2] - seg[0];
          double dy = seg[3] - seg[1];
          double angle_rad = std::atan2(std::abs(dy), std::abs(dx));
          if (angle_rad >= min_theta_rad && angle_rad <= max_theta_rad) {
            filtered.push_back(seg);
          }
        }
        segments_out = std::move(filtered);
      }
    }
  }

  void publish_lines(const std::vector<cv::Vec4i>& segments,
                     const cv::Rect& roi_rect) {
    auto lines_msg = std_msgs::msg::Float32MultiArray();
    lines_msg.layout.dim.resize(2);
    lines_msg.layout.dim[0].label = "lines";
    lines_msg.layout.dim[0].size = segments.size();
    lines_msg.layout.dim[0].stride = segments.size() * 4;
    lines_msg.layout.dim[1].label = "points";
    lines_msg.layout.dim[1].size = 4;
    lines_msg.layout.dim[1].stride = 4;

    for (const auto& seg : segments) {
      // Map to full image coordinates
      lines_msg.data.push_back(static_cast<float>(seg[0] + roi_rect.x));
      lines_msg.data.push_back(static_cast<float>(seg[1] + roi_rect.y));
      lines_msg.data.push_back(static_cast<float>(seg[2] + roi_rect.x));
      lines_msg.data.push_back(static_cast<float>(seg[3] + roi_rect.y));
    }
    lines_pub_->publish(lines_msg);
  }

  void publish_visualization(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                             const cv::Mat& original_img,
                             const cv::Mat& work_img, const cv::Mat& edges,
                             const std::vector<cv::Vec4i>& segments,
                             const cv::Rect& roi_rect) {
    cv::Mat display;
    if (show_edges_) {
      // Show edges instead of lines
      cv::Mat edges_color;
      cv::cvtColor(edges, edges_color, cv::COLOR_GRAY2BGR);
      display = original_img.clone();
      edges_color.copyTo(display(roi_rect));
    } else {
      // Draw lines on the image
      display = original_img.clone();
      const cv::Scalar color(static_cast<int>(draw_color_bgr_[0]),
                             static_cast<int>(draw_color_bgr_[1]),
                             static_cast<int>(draw_color_bgr_[2]));
      for (const auto& seg : segments) {
        cv::line(display, cv::Point(seg[0] + roi_rect.x, seg[1] + roi_rect.y),
                 cv::Point(seg[2] + roi_rect.x, seg[3] + roi_rect.y), color,
                 draw_thickness_);
      }
    }

    // Draw calibration visualization if in calibration state
    if (state_ == State::Calibrating && calibrator_->has_valid_ellipse()) {
      cv::RotatedRect ellipse = calibrator_->get_last_ellipse();
      cv::ellipse(display, ellipse, cv::Scalar(255, 0, 255), 2);
      if (calibrator_->has_valid_circle()) {
        cv::Point2d circle = calibrator_->get_last_circle();
        cv::circle(
            display,
            cv::Point(static_cast<int>(circle.x), static_cast<int>(circle.y)),
            5, cv::Scalar(0, 255, 255), -1);
      }
    }

    // Draw localization visualization
    if (state_ == State::Localizing && localization_valid_) {
      // Draw robot pose info on the image
      std::string pose_text =
          cv::format("x:%.2f y:%.2f yaw:%.1f", last_robot_x_, last_robot_y_,
                     rad2deg(last_robot_yaw_));
      cv::putText(display, pose_text, cv::Point(10, 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    // Publish
    sensor_msgs::msg::Image::SharedPtr out_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8,
                           display)
            .toImageMsg();
    image_pub_->publish(*out_msg);
  }

 private:
  // ===== Localization: robot pose estimation using landmarks =====
  void perform_localization(const std::vector<cv::Vec4i>& segments,
                            const cv::Mat& img) {
    if (!has_cam_info_ || !std::isfinite(estimated_pitch_rad_)) {
      RCLCPP_WARN(this->get_logger(),
                  "Localization skipped: cam_info=%s, pitch_valid=%s",
                  has_cam_info_ ? "true" : "false",
                  std::isfinite(estimated_pitch_rad_) ? "true" : "false");
      return;
    }

    // Step 1: Detect gray disk landmark (reuse calibration detection)
    double landmark_x_px = 0.0, landmark_y_px = 0.0;
    bool landmark_found = false;

    // Use same detection logic as calibration but without ROI restriction
    cv::Rect full_roi(0, 0, img.cols, img.rows);
    if (calibrator_->detect_landmark_center(img, full_roi, 1.0, landmark_x_px,
                                            landmark_y_px)) {
      landmark_found = true;
    } else {
      RCLCPP_WARN(this->get_logger(), "Landmark not found");
    }

    // Step 2: Find the front line (closest to camera, nearly vertical)
    cv::Vec4i front_line;
    bool front_line_found = find_front_line(segments, front_line);

    if (!front_line_found) {
      RCLCPP_WARN(this->get_logger(), "Front line not found (total lines: %zu)",
                  segments.size());
    }

    // Step 3: Calculate robot pose if both landmarks are detected
    if (landmark_found && front_line_found) {
      double robot_x, robot_y, robot_yaw;
      if (calculate_robot_pose(landmark_x_px, landmark_y_px, front_line,
                               robot_x, robot_y, robot_yaw)) {
        // Store for visualization (no logging here)
        last_robot_x_ = robot_x;
        last_robot_y_ = robot_y;
        last_robot_yaw_ = robot_yaw;
        localization_valid_ = true;
      } else {
        RCLCPP_WARN(this->get_logger(), "Failed to calculate robot pose");
      }
    } else {
      localization_valid_ = false;
    }
  }

  bool find_front_line(const std::vector<cv::Vec4i>& segments,
                       cv::Vec4i& front_line) {
    if (segments.empty()) return false;

    // Find the line closest to the camera (lowest y-coordinate)
    // and most vertical (angle close to 90 degrees)
    double best_score = std::numeric_limits<double>::infinity();
    bool found = false;

    for (const auto& line : segments) {
      // Calculate line properties
      double dx = line[2] - line[0];
      double dy = line[3] - line[1];
      double length = std::sqrt(dx * dx + dy * dy);

      if (length < 50.0) continue;  // Skip very short lines

      // Calculate angle from horizontal (0 = horizontal, 90 = vertical)
      double angle_rad = std::atan2(std::abs(dx), std::abs(dy));
      double angle_deg = rad2deg(angle_rad);

      // Prefer nearly vertical lines (80-90 degrees)
      if (angle_deg < 70.0) continue;

      // Calculate distance to camera (use minimum y-coordinate)
      double min_y = std::min(line[1], line[3]);

      // Score: prefer closer lines (lower y) and more vertical lines
      double angle_penalty = std::abs(90.0 - angle_deg) * 0.5;
      double distance_penalty = min_y * 0.01;
      double score = distance_penalty + angle_penalty;

      if (score < best_score) {
        best_score = score;
        front_line = line;
        found = true;
      }
    }

    return found;
  }

  bool calculate_robot_pose(double landmark_px_x, double landmark_px_y,
                            const cv::Vec4i& front_line, double& robot_x,
                            double& robot_y, double& robot_yaw) {
    // Transform landmark pixel coordinates to world coordinates
    // Using the calibrated camera model and known landmark position

    // Step 1: Convert landmark pixels to camera ray
    double u_landmark = (landmark_px_x - cx_) / fx_;
    double v_landmark = (landmark_px_y - cy_) / fy_;

    // Step 2: Project ray to ground plane using estimated pitch
    // The pitch is defined as rotation around x-axis (positive = looking up,
    // negative = looking down)
    double cos_pitch = std::cos(estimated_pitch_rad_);
    double sin_pitch = std::sin(estimated_pitch_rad_);

    // Ray direction in camera frame (z forward, y down, x right)
    // For standard pinhole camera model: u = X/Z, v = Y/Z where (X,Y,Z) is 3D
    // point in camera frame
    double ray_x = u_landmark;  // X/Z
    double ray_y = v_landmark;  // Y/Z
    double ray_z = 1.0;         // Z/Z (normalized)

    // Transform ray from camera frame to world frame
    // Camera frame to world frame with correct orientation:
    // Camera: z forward (into scene), y down, x right
    // World: z down (toward ground), y forward, x right
    // The camera is tilted down by |pitch| degrees

    // Corrected transformation: map camera z to world -z (downward)
    double world_ray_x = ray_x;
    double world_ray_y = ray_y * cos_pitch + ray_z * sin_pitch;
    double world_ray_z =
        -ray_y * sin_pitch - ray_z * cos_pitch;  // Map to downward direction

    // Intersect with ground plane (z = -camera_height_m_)
    if (std::abs(world_ray_z) < 1e-6) {
      return false;  // Ray parallel to ground
    }

    double t = -camera_height_m_ / world_ray_z;
    if (t <= 0) {
      return false;  // Ray pointing away from ground
    }

    double landmark_world_x = t * world_ray_x;
    double landmark_world_y = t * world_ray_y;

    // Step 3: Calculate front line orientation
    double line_dx = front_line[2] - front_line[0];
    double line_dy = front_line[3] - front_line[1];
    double line_angle_image = std::atan2(
        -line_dy, line_dx);  // Negative dy because image y increases downward

    // The front line represents the map y-axis, so robot yaw is the angle of
    // this line
    robot_yaw = line_angle_image;

    // Step 4: Calculate robot position
    // Robot position is landmark map position minus camera-relative landmark
    // position Rotated by robot yaw angle
    double cos_yaw = std::cos(robot_yaw);
    double sin_yaw = std::sin(robot_yaw);

    // Transform landmark offset from robot frame to map frame
    double offset_x = -landmark_world_x * cos_yaw + landmark_world_y * sin_yaw;
    double offset_y = -landmark_world_x * sin_yaw - landmark_world_y * cos_yaw;

    robot_x = landmark_map_x_ + offset_x;
    robot_y = landmark_map_y_ + offset_y;

    return true;
  }

  // Parameters
  std::string image_topic_;
  std::string camera_info_topic_;
  bool publish_image_{};
  bool show_edges_{};
  bool use_hsv_mask_{};
  int blur_ksize_{};
  double blur_sigma_{};
  std::vector<int64_t> roi_{};  // x,y,w,h (-1 = disabled)

  int canny_low_{};
  int canny_high_{};
  int canny_aperture_{};
  bool canny_L2gradient_{};
  std::string hough_type_{};  // "probabilistic" or "standard"
  double rho_{};
  double theta_deg_{};
  int threshold_{};
  double min_line_length_{};
  double max_line_gap_{};
  double min_theta_deg_{};
  double max_theta_deg_{};
  std::vector<int64_t> draw_color_bgr_{};
  int draw_thickness_{};

  // HSV mask parameters
  int hsv_lower_h_{};
  int hsv_lower_s_{};
  int hsv_lower_v_{};
  int hsv_upper_h_{};
  int hsv_upper_s_{};
  int hsv_upper_v_{};
  int hsv_dilate_kernel_{};
  int hsv_dilate_iter_{};

  // Camera model (for localization)
  State state_{State::Calibrating};
  bool has_cam_info_{false};
  double fx_{};
  double fy_{};
  double cx_{};
  double cy_{};

  // Calibration parameters (kept for CameraCalibrator access)
  double camera_height_m_{};
  double landmark_distance_m_{};
  double calib_timeout_sec_{};
  std::vector<int64_t> calib_roi_{};
  int calib_hsv_s_max_{};
  int calib_hsv_v_min_{};
  int calib_hsv_v_max_{};
  int calib_min_area_{};
  int calib_min_major_px_{};
  double calib_max_major_ratio_{};
  double calib_fill_min_{};

  // Localization parameters
  double landmark_map_x_{};
  double landmark_map_y_{};

  // Localization state
  bool localization_valid_{false};
  double last_robot_x_{};
  double last_robot_y_{};
  double last_robot_yaw_{};

  double estimated_pitch_rad_{std::numeric_limits<double>::quiet_NaN()};

  std::unique_ptr<CameraCalibrator> calibrator_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lines_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      param_cb_handle_;

  std::mutex param_mutex_;
};

// ===== CameraCalibrator Implementation =====
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
    // We prefer horizontal major axis (squashed vertically) -> angle near 0/180
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

  if (!node_->has_cam_info_) return;
  if (v_samples_.size() < kMinCalibSamples) return;

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

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LineDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
