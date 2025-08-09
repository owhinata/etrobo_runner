// Minimal single-file implementation per doc/DESIGN.md

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>

using std::placeholders::_1;

class LineDetectorNode : public rclcpp::Node {
 public:
  LineDetectorNode() : rclcpp::Node("etrobo_line_detector") {
    // Declare parameters with defaults
    image_topic_ = this->declare_parameter<std::string>("image_topic", "image");
    use_color_output_ = this->declare_parameter<bool>("use_color_output", true);

    grayscale_ = this->declare_parameter<bool>("grayscale", true);
    use_hsv_mask_ = this->declare_parameter<bool>("use_hsv_mask", true);
    blur_ksize_ = this->declare_parameter<int>("blur_ksize", 5);
    blur_sigma_ = this->declare_parameter<double>("blur_sigma", 1.5);
    roi_ = this->declare_parameter<std::vector<int64_t>>(
        "roi", std::vector<int64_t>{-1, -1, -1, -1});
    downscale_ = this->declare_parameter<double>("downscale", 1.0);

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
    publish_markers_ = this->declare_parameter<bool>("publish_markers", true);

    // HSV mask parameters
    hsv_lower_h_ = this->declare_parameter<int>("hsv_lower_h", 0);
    hsv_lower_s_ = this->declare_parameter<int>("hsv_lower_s", 0);
    hsv_lower_v_ = this->declare_parameter<int>("hsv_lower_v", 0);
    hsv_upper_h_ = this->declare_parameter<int>("hsv_upper_h", 180);
    hsv_upper_s_ = this->declare_parameter<int>("hsv_upper_s", 120);
    hsv_upper_v_ = this->declare_parameter<int>("hsv_upper_v", 150);
    hsv_dilate_kernel_ = this->declare_parameter<int>("hsv_dilate_kernel", 3);
    hsv_dilate_iter_ = this->declare_parameter<int>("hsv_dilate_iter", 1);

    // Validate parameters minimally
    sanitize_parameters();

    // Publishers
    // Use SensorData QoS but set RELIABLE to interoperate with image_view
    // (subscriber expects reliable reliability).
    auto pub_qos = rclcpp::SensorDataQoS();
    pub_qos.reliable();
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "image_with_lines", pub_qos);
    lines_pub_ =
        this->create_publisher<std_msgs::msg::Float32MultiArray>("lines", 10);
    if (publish_markers_) {
      markers_pub_ =
          this->create_publisher<visualization_msgs::msg::MarkerArray>(
              "markers", 10);
    }

    // Subscription with SensorDataQoS depth=1, best effort
    auto qos = rclcpp::SensorDataQoS();
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, qos,
        std::bind(&LineDetectorNode::image_callback, this, _1));

    // Dynamic parameter updates for processing params (not topic/QoS)
    param_cb_handle_ = this->add_on_set_parameters_callback(std::bind(
        &LineDetectorNode::on_parameters_set, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "Line detector node initialized. Subscribing to: %s",
                image_topic_.c_str());
  }

 private:
  void sanitize_parameters() {
    if (blur_ksize_ <= 0) blur_ksize_ = 1;
    if (blur_ksize_ % 2 == 0) blur_ksize_ += 1;  // must be odd
    if (downscale_ <= 0.0) downscale_ = 1.0;
    if (canny_low_ < 0) canny_low_ = 0;
    if (canny_high_ < 0) canny_high_ = 0;
    if (canny_aperture_ != 3 && canny_aperture_ != 5 && canny_aperture_ != 7) {
      canny_aperture_ = 3;
    }
    if (theta_deg_ <= 0.0) theta_deg_ = 1.0;
    if (threshold_ < 1) threshold_ = 1;
    if (min_theta_deg_ < 0.0) min_theta_deg_ = 0.0;
    if (max_theta_deg_ > 180.0) max_theta_deg_ = 180.0;
    if (min_theta_deg_ > max_theta_deg_)
      std::swap(min_theta_deg_, max_theta_deg_);
    if (draw_color_bgr_.size() != 3) draw_color_bgr_ = {0, 255, 0};
    // HSV boundaries and morphology
    auto clamp = [](int v, int lo, int hi) {
      return std::max(lo, std::min(v, hi));
    };
    hsv_lower_h_ = clamp(hsv_lower_h_, 0, 180);
    hsv_upper_h_ = clamp(hsv_upper_h_, 0, 180);
    hsv_lower_s_ = clamp(hsv_lower_s_, 0, 255);
    hsv_upper_s_ = clamp(hsv_upper_s_, 0, 255);
    hsv_lower_v_ = clamp(hsv_lower_v_, 0, 255);
    hsv_upper_v_ = clamp(hsv_upper_v_, 0, 255);
    if (hsv_lower_h_ > hsv_upper_h_) std::swap(hsv_lower_h_, hsv_upper_h_);
    if (hsv_lower_s_ > hsv_upper_s_) std::swap(hsv_lower_s_, hsv_upper_s_);
    if (hsv_lower_v_ > hsv_upper_v_) std::swap(hsv_lower_v_, hsv_upper_v_);
    if (hsv_dilate_kernel_ < 1) hsv_dilate_kernel_ = 1;
    if ((hsv_dilate_kernel_ % 2) == 0) hsv_dilate_kernel_ += 1;  // make odd
    if (hsv_dilate_iter_ < 0) hsv_dilate_iter_ = 0;
  }

  rcl_interfaces::msg::SetParametersResult on_parameters_set(
      const std::vector<rclcpp::Parameter> &params) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "ok";

    for (const auto &p : params) {
      const auto &name = p.get_name();
      try {
        if (name == "use_color_output")
          use_color_output_ = p.as_bool();
        else if (name == "grayscale")
          grayscale_ = p.as_bool();
        else if (name == "use_hsv_mask")
          use_hsv_mask_ = p.as_bool();
        else if (name == "blur_ksize")
          blur_ksize_ = p.as_int();
        else if (name == "blur_sigma")
          blur_sigma_ = p.as_double();
        else if (name == "roi")
          roi_ = p.as_integer_array();
        else if (name == "downscale")
          downscale_ = p.as_double();
        else if (name == "canny_low")
          canny_low_ = p.as_int();
        else if (name == "canny_high")
          canny_high_ = p.as_int();
        else if (name == "canny_aperture")
          canny_aperture_ = p.as_int();
        else if (name == "canny_L2gradient")
          canny_L2gradient_ = p.as_bool();
        else if (name == "hough_type")
          hough_type_ = p.as_string();
        else if (name == "rho")
          rho_ = p.as_double();
        else if (name == "theta_deg")
          theta_deg_ = p.as_double();
        else if (name == "threshold")
          threshold_ = p.as_int();
        else if (name == "min_line_length")
          min_line_length_ = p.as_double();
        else if (name == "max_line_gap")
          max_line_gap_ = p.as_double();
        else if (name == "min_theta_deg")
          min_theta_deg_ = p.as_double();
        else if (name == "max_theta_deg")
          max_theta_deg_ = p.as_double();
        else if (name == "draw_color_bgr") {
          auto v = p.as_integer_array();
          draw_color_bgr_.assign(v.begin(), v.end());
        } else if (name == "draw_thickness")
          draw_thickness_ = p.as_int();
        else if (name == "publish_markers")
          publish_markers_ = p.as_bool();
        // HSV mask params
        else if (name == "hsv_lower_h")
          hsv_lower_h_ = p.as_int();
        else if (name == "hsv_lower_s")
          hsv_lower_s_ = p.as_int();
        else if (name == "hsv_lower_v")
          hsv_lower_v_ = p.as_int();
        else if (name == "hsv_upper_h")
          hsv_upper_h_ = p.as_int();
        else if (name == "hsv_upper_s")
          hsv_upper_s_ = p.as_int();
        else if (name == "hsv_upper_v")
          hsv_upper_v_ = p.as_int();
        else if (name == "hsv_dilate_kernel")
          hsv_dilate_kernel_ = p.as_int();
        else if (name == "hsv_dilate_iter")
          hsv_dilate_iter_ = p.as_int();
        else if (name == "image_topic") {
          // Do not allow changing topic at runtime as it requires
          // resubscription
          result.successful = false;
          result.reason = "Changing image_topic at runtime is not supported";
        }
      } catch (const std::exception &e) {
        result.successful = false;
        result.reason = e.what();
      }
    }

    sanitize_parameters();
    return result;
  }

  static cv::Rect valid_roi(const cv::Mat &img,
                            const std::vector<int64_t> &roi) {
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

  static std::vector<cv::Vec4i> hough_standard_to_segments(
      const std::vector<cv::Vec2f> &lines, int width, int height,
      double min_theta_rad, double max_theta_rad) {
    std::vector<cv::Vec4i> segments;
    // Image rectangle
    const cv::Rect rect(0, 0, width, height);
    for (const auto &l : lines) {
      float rho = l[0];
      float theta = l[1];
      if (theta < min_theta_rad || theta > max_theta_rad) continue;

      double a = std::cos(theta), b = std::sin(theta);
      double x0 = a * rho, y0 = b * rho;
      // Compute two points far along the line
      cv::Point2f pt1, pt2;
      pt1.x = static_cast<float>(x0 + 10000 * (-b));
      pt1.y = static_cast<float>(y0 + 10000 * (a));
      pt2.x = static_cast<float>(x0 - 10000 * (-b));
      pt2.y = static_cast<float>(y0 - 10000 * (a));

      // Clip the segment to image rectangle
      std::vector<cv::Point2f> points = {pt1, pt2};
      // Liang–Barsky would be ideal; OpenCV clipLine works for ints
      cv::Point p1(cvRound(pt1.x), cvRound(pt1.y));
      cv::Point p2(cvRound(pt2.x), cvRound(pt2.y));
      if (cv::clipLine(rect, p1, p2)) {
        segments.emplace_back(cv::Vec4i(p1.x, p1.y, p2.x, p2.y));
      }
    }
    return segments;
  }

  void publish_markers(const std::vector<cv::Vec4i> &lines,
                       const std_msgs::msg::Header &header) {
    if (!publish_markers_ || !markers_pub_) return;
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker m;
    m.header = header;
    m.ns = "lines";
    m.type = visualization_msgs::msg::Marker::LINE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = std::max(0.001, static_cast<double>(draw_thickness_));
    m.color.a = 1.0;
    m.color.b = static_cast<double>(draw_color_bgr_[0]) / 255.0;
    m.color.g = static_cast<double>(draw_color_bgr_[1]) / 255.0;
    m.color.r = static_cast<double>(draw_color_bgr_[2]) / 255.0;
    m.id = 0;
    m.pose.orientation.w = 1.0;

    m.points.reserve(lines.size() * 2);
    for (const auto &l : lines) {
      geometry_msgs::msg::Point p1, p2;
      p1.x = l[0];
      p1.y = l[1];
      p1.z = 0.0;
      p2.x = l[2];
      p2.y = l[3];
      p2.z = 0.0;
      m.points.push_back(p1);
      m.points.push_back(p2);
    }
    array.markers.push_back(m);
    markers_pub_->publish(array);
  }

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    const auto t0 = std::chrono::steady_clock::now();
    // Convert to cv::Mat (try BGR8)
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      if (msg->encoding == sensor_msgs::image_encodings::BGR8 ||
          msg->encoding == sensor_msgs::image_encodings::RGB8 ||
          msg->encoding == sensor_msgs::image_encodings::MONO8) {
        cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
      } else {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      }
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat img;
    if (cv_ptr->encoding == sensor_msgs::image_encodings::RGB8) {
      cv::cvtColor(cv_ptr->image, img, cv::COLOR_RGB2BGR);
    } else if (cv_ptr->encoding == sensor_msgs::image_encodings::BGR8) {
      img = cv_ptr->image;
    } else if (cv_ptr->encoding == sensor_msgs::image_encodings::MONO8) {
      img = cv_ptr->image;
    } else {
      // already converted to BGR8 above
      img = cv_ptr->image;
    }

    // ROI
    cv::Rect roi_rect = valid_roi(img, roi_);
    cv::Mat work = img(roi_rect).clone();

    // Downscale
    double scale = 1.0;
    if (downscale_ != 1.0) {
      scale = std::max(1e-6, downscale_);
      cv::Mat tmp;
      cv::resize(work, tmp, cv::Size(), 1.0 / scale, 1.0 / scale,
                 cv::INTER_AREA);
      work = tmp;
    }

    // Grayscale and blur
    cv::Mat gray;
    if (grayscale_) {
      if (work.channels() == 3) {
        cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
      } else {
        gray = work;
      }
    } else {
      // operate on one channel anyway for Canny
      if (work.channels() == 3) {
        cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
      } else {
        gray = work;
      }
    }
    if (blur_ksize_ > 1) {
      cv::GaussianBlur(gray, gray, cv::Size(blur_ksize_, blur_ksize_),
                       blur_sigma_);
    }

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
      const cv::Scalar lower(0, 0, 0);
      const cv::Scalar upper(180, 120, 150);  // wider S/V to keep near-black
      cv::Mat mask;
      cv::inRange(hsv_src, lower, upper, mask);
      // Thicken mask slightly so boundary edges survive masking
      cv::Mat kernel =
          cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 1);
      cv::bitwise_and(edges, mask, edges);
    }

    // Hough
    std::vector<cv::Vec4i> segments;
    if (hough_type_ == "standard") {
      std::vector<cv::Vec2f> lines;
      double theta = theta_deg_ * CV_PI / 180.0;
      cv::HoughLines(edges, lines, rho_, theta, threshold_);
      double min_tr = min_theta_deg_ * CV_PI / 180.0;
      double max_tr = max_theta_deg_ * CV_PI / 180.0;
      segments = hough_standard_to_segments(lines, edges.cols, edges.rows,
                                            min_tr, max_tr);
    } else {
      // probabilistic default
      cv::HoughLinesP(edges, segments, rho_, theta_deg_ * CV_PI / 180.0,
                      threshold_, min_line_length_, max_line_gap_);
    }

    // Restore coordinates to original scale and ROI
    std::vector<cv::Vec4i> segments_full;
    segments_full.reserve(segments.size());
    for (const auto &l : segments) {
      int x1 = static_cast<int>(std::round(l[0] * scale)) + roi_rect.x;
      int y1 = static_cast<int>(std::round(l[1] * scale)) + roi_rect.y;
      int x2 = static_cast<int>(std::round(l[2] * scale)) + roi_rect.x;
      int y2 = static_cast<int>(std::round(l[3] * scale)) + roi_rect.y;
      segments_full.emplace_back(cv::Vec4i{x1, y1, x2, y2});
    }

    // Visualization image
    cv::Mat vis;
    if (use_color_output_) {
      if (img.channels() == 1) {
        cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
      } else {
        vis = img.clone();
      }
    } else {
      if (img.channels() == 3) {
        cv::cvtColor(img, vis, cv::COLOR_BGR2GRAY);
      } else {
        vis = img.clone();
      }
    }
    for (const auto &l : segments_full) {
      cv::line(vis, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
               cv::Scalar(static_cast<int>(draw_color_bgr_[0]),
                          static_cast<int>(draw_color_bgr_[1]),
                          static_cast<int>(draw_color_bgr_[2])),
               draw_thickness_);
    }

    // Publish image
    std_msgs::msg::Header header = msg->header;
    cv_bridge::CvImage out_img;
    out_img.header = header;
    out_img.encoding = use_color_output_ ? sensor_msgs::image_encodings::BGR8
                                         : sensor_msgs::image_encodings::MONO8;
    out_img.image = vis;
    image_pub_->publish(*out_img.toImageMsg());

    // Publish lines
    std_msgs::msg::Float32MultiArray lines_msg;
    lines_msg.layout.dim.resize(1);
    lines_msg.layout.dim[0].label = "lines_flat_xyxy";
    lines_msg.layout.dim[0].size = segments_full.size() * 4;
    lines_msg.layout.dim[0].stride = 1;
    lines_msg.data.reserve(segments_full.size() * 4);
    for (const auto &l : segments_full) {
      lines_msg.data.push_back(static_cast<float>(l[0]));
      lines_msg.data.push_back(static_cast<float>(l[1]));
      lines_msg.data.push_back(static_cast<float>(l[2]));
      lines_msg.data.push_back(static_cast<float>(l[3]));
    }
    lines_pub_->publish(lines_msg);

    // Publish markers (optional)
    publish_markers(segments_full, header);

    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0)
            .count();
    RCLCPP_INFO(this->get_logger(), "Processed frame: %zu lines in %.2f ms",
                segments_full.size(), ms);
  }

  // Parameters
  std::string image_topic_;
  bool use_color_output_{};
  bool grayscale_{};
  bool use_hsv_mask_{};
  int blur_ksize_{};
  double blur_sigma_{};
  std::vector<int64_t> roi_{};  // x,y,w,h (-1 = disabled)
  double downscale_{};
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
  bool publish_markers_{};

  // HSV mask parameters
  int hsv_lower_h_{};
  int hsv_lower_s_{};
  int hsv_lower_v_{};
  int hsv_upper_h_{};
  int hsv_upper_s_{};
  int hsv_upper_v_{};
  int hsv_dilate_kernel_{};
  int hsv_dilate_iter_{};

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lines_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      markers_pub_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  std::mutex param_mutex_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LineDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
