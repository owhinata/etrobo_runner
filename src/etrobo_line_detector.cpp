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
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>

using std::placeholders::_1;

class LineDetectorNode : public rclcpp::Node {
 public:
  LineDetectorNode() : rclcpp::Node("etrobo_line_detector") {
    declare_parameters();
    sanitize_parameters();
    setup_publishers();
    setup_subscription();
    setup_camera_info_subscription();
    setup_parameter_callback();

    RCLCPP_INFO(this->get_logger(),
                "Line detector node initialized. Subscribing to: %s",
                image_topic_.c_str());
  }

 private:
  void declare_parameters() {
    image_topic_ = this->declare_parameter<std::string>("image_topic", "image");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "camera_info");
    use_color_output_ = this->declare_parameter<bool>("use_color_output", true);
    publish_image_ =
        this->declare_parameter<bool>("publish_image_with_lines", false);

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

    // Edge closing parameters
    use_edge_close_ = this->declare_parameter<bool>("use_edge_close", true);
    edge_close_kernel_ = this->declare_parameter<int>("edge_close_kernel", 3);
    edge_close_iter_ = this->declare_parameter<int>("edge_close_iter", 1);

    // Temporal smoothing (EMA) parameters
    enable_temporal_smoothing_ =
        this->declare_parameter<bool>("enable_temporal_smoothing", true);
    ema_alpha_ = this->declare_parameter<double>("ema_alpha", 0.5);
    match_max_px_ = this->declare_parameter<int>("match_max_px", 20);
    match_max_angle_deg_ =
        this->declare_parameter<double>("match_max_angle_deg", 10.0);
    min_age_to_publish_ = this->declare_parameter<int>("min_age_to_publish", 2);
    max_missed_ = this->declare_parameter<int>("max_missed", 3);

    // Calibration parameters
    camera_height_m_ =
        this->declare_parameter<double>("camera_height_meters", 0.2);
    landmark_distance_m_ =
        this->declare_parameter<double>("landmark_distance_meters", 0.59);
    calib_timeout_sec_ =
        this->declare_parameter<double>("calib_timeout_sec", 60.0);
    calib_roi_ = this->declare_parameter<std::vector<int64_t>>(
        "calib_roi", std::vector<int64_t>{-1, -1, -1, -1});
    calib_hsv_s_max_ = this->declare_parameter<int>("calib_hsv_s_max", 60);
    calib_hsv_v_min_ = this->declare_parameter<int>("calib_hsv_v_min", 80);
    calib_hsv_v_max_ = this->declare_parameter<int>("calib_hsv_v_max", 168);
    calib_min_area_ = this->declare_parameter<int>("calib_min_area", 80);
    calib_min_major_px_ = this->declare_parameter<int>("calib_min_major_px", 8);
    calib_max_major_ratio_ =
        this->declare_parameter<double>("calib_max_major_ratio", 0.65);
    calib_fill_min_ = this->declare_parameter<double>("calib_fill_min", 0.25);
  }

  void setup_publishers() {
    // Use SensorData QoS but set RELIABLE to interoperate with image_view
    // (subscriber expects reliable reliability).
    auto pub_qos = rclcpp::SensorDataQoS();
    pub_qos.reliable();
    if (publish_image_) {
      image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
          "image_with_lines", pub_qos);
    }
    lines_pub_ =
        this->create_publisher<std_msgs::msg::Float32MultiArray>("lines", 10);
    if (publish_markers_) {
      markers_pub_ =
          this->create_publisher<visualization_msgs::msg::MarkerArray>(
              "markers", 10);
    }
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

  cv::Mat preprocess_image(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                           cv::Mat &original_img, cv::Rect &roi_rect,
                           double &scale) {
    // Convert to cv::Mat
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
      return cv::Mat();
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

    original_img = img;

    // ROI
    roi_rect = valid_roi(img, roi_);
    cv::Mat work = img(roi_rect).clone();

    // Downscale
    scale = 1.0;
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

    return gray;
  }

  cv::Mat detect_edges(const cv::Mat &gray, const cv::Mat &work) {
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
    // Optional edge closing to connect broken edges before Hough
    if (use_edge_close_ && edge_close_iter_ > 0) {
      cv::Mat kernel = cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(edge_close_kernel_, edge_close_kernel_));
      for (int i = 0; i < edge_close_iter_; ++i) {
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);
      }
    }

    return edges;
  }

  std::vector<cv::Vec4i> detect_lines(const cv::Mat &edges) {
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
    return segments;
  }

  std::vector<cv::Vec4i> restore_coordinates(
      const std::vector<cv::Vec4i> &segments, const cv::Rect &roi_rect,
      double scale) {
    std::vector<cv::Vec4i> segments_full;
    segments_full.reserve(segments.size());
    for (const auto &l : segments) {
      int x1 = static_cast<int>(std::round(l[0] * scale)) + roi_rect.x;
      int y1 = static_cast<int>(std::round(l[1] * scale)) + roi_rect.y;
      int x2 = static_cast<int>(std::round(l[2] * scale)) + roi_rect.x;
      int y2 = static_cast<int>(std::round(l[3] * scale)) + roi_rect.y;
      segments_full.emplace_back(cv::Vec4i{x1, y1, x2, y2});
    }
    return segments_full;
  }

  void publish_visualization(const std::vector<cv::Vec4i> &segments,
                             const cv::Mat &original_img,
                             const std_msgs::msg::Header &header) {
    if (!image_pub_) return;

    cv::Mat vis;
    if (use_color_output_) {
      if (original_img.channels() == 1) {
        cv::cvtColor(original_img, vis, cv::COLOR_GRAY2BGR);
      } else {
        vis = original_img.clone();
      }
    } else {
      if (original_img.channels() == 3) {
        cv::cvtColor(original_img, vis, cv::COLOR_BGR2GRAY);
      } else {
        vis = original_img.clone();
      }
    }

    for (const auto &l : segments) {
      cv::line(vis, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
               cv::Scalar(static_cast<int>(draw_color_bgr_[0]),
                          static_cast<int>(draw_color_bgr_[1]),
                          static_cast<int>(draw_color_bgr_[2])),
               draw_thickness_);
    }

    // Calibration overlay: draw detected gray ellipse/circle and line from
    // camera ground foot
    if (last_circle_valid_ || last_ellipse_valid_) {
      cv::Scalar circle_color(0, 255, 255);  // yellow
      cv::Point c_pt(static_cast<int>(std::lround(last_circle_px_.x)),
                     static_cast<int>(std::lround(last_circle_px_.y)));
      if (last_ellipse_valid_) {
        cv::ellipse(vis, last_ellipse_full_, circle_color, 2);
        c_pt = cv::Point(
            static_cast<int>(std::lround(last_ellipse_full_.center.x)),
            static_cast<int>(std::lround(last_ellipse_full_.center.y)));
      }
      if (c_pt.x >= 0 && c_pt.y >= 0 && c_pt.x < vis.cols &&
          c_pt.y < vis.rows) {
        cv::circle(vis, c_pt, 6, circle_color, 2);
      }
      // Annotate selected candidate metrics (a/r/V/S)
      if (last_ellipse_valid_) {
        char txt2[80];
        std::snprintf(txt2, sizeof(txt2), "a=%.0f r=%.2f v=%.0f s=%.0f",
                      last_angle_deg_, last_ratio_, last_mean_v_, last_mean_s_);
        cv::Point t2p(c_pt.x + 8, c_pt.y - 8);
        cv::putText(vis, txt2, t2p, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);
      }
      if (has_cam_info_ && std::isfinite(estimated_pitch_rad_)) {
        // Compute ground foot projection pixel
        double tan_theta = std::tan(estimated_pitch_rad_);
        if (std::abs(tan_theta) > 1e-6) {
          double v0 = cy_ - fy_ / tan_theta;  // row of camera ground foot
          double u0 = cx_;                    // column at principal point
          cv::Point p0(static_cast<int>(std::lround(u0)),
                       static_cast<int>(std::lround(v0)));
          cv::Point p1 = c_pt;
          cv::Rect rect(0, 0, vis.cols, vis.rows);
          cv::Point p0c = p0, p1c = p1;
          if (cv::clipLine(rect, p0c, p1c)) {
            cv::line(vis, p0c, p1c, cv::Scalar(255, 200, 0), 2);
            cv::circle(vis, p0c, 4, cv::Scalar(0, 128, 255), -1);
          }
          // Overlay text with distance and pitch
          char buf[128];
          std::snprintf(buf, sizeof(buf), "D=%.2fm, pitch=%.2f deg",
                        landmark_distance_m_,
                        estimated_pitch_rad_ * 180.0 / CV_PI);
          int baseline = 0;
          cv::Size txt =
              cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
          cv::Point org(10, 20 + txt.height);
          cv::rectangle(vis, org + cv::Point(0, baseline),
                        org + cv::Point(txt.width, -txt.height),
                        cv::Scalar(0, 0, 0), cv::FILLED);
          cv::putText(vis, buf, org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                      cv::Scalar(0, 255, 255), 2);
        }
      }
    }

    // Draw top calibration candidates (debug) during calibration
    if (state_ == State::CalibratePitch && !calib_cands_.empty()) {
      std::vector<CalibCand> cands = calib_cands_;
      std::sort(cands.begin(), cands.end(),
                [](const CalibCand &a, const CalibCand &b) {
                  return a.score < b.score;
                });
      const int max_draw = std::min<int>(static_cast<int>(cands.size()), 8);
      for (int i = 0; i < max_draw; ++i) {
        const auto &cc = cands[i];
        cv::Scalar col(255, 200, 200);
        cv::ellipse(vis, cc.ellipse, col, 1);
        char txt[64];
        std::snprintf(txt, sizeof(txt), "a=%.0f r=%.2f v=%.0f s=%.0f",
                      cc.angle_deg, cc.ratio, cc.mean_v, cc.mean_s);
        cv::Point tp(static_cast<int>(std::lround(cc.ellipse.center.x)) + 6,
                     static_cast<int>(std::lround(cc.ellipse.center.y)) - 6);
        if (tp.x >= 0 && tp.y >= 0 && tp.x < vis.cols && tp.y < vis.rows) {
          cv::putText(vis, txt, tp, cv::FONT_HERSHEY_SIMPLEX, 0.4, col, 1);
        }
      }
    }

    // Show calibration ROI and status text during calibration for debugging
    if (state_ == State::CalibratePitch) {
      // Draw calib ROI if provided
      if (calib_roi_.size() == 4 && calib_roi_[0] >= 0 && calib_roi_[1] >= 0 &&
          calib_roi_[2] > 0 && calib_roi_[3] > 0) {
        cv::Rect cr(
            static_cast<int>(calib_roi_[0]), static_cast<int>(calib_roi_[1]),
            static_cast<int>(calib_roi_[2]), static_cast<int>(calib_roi_[3]));
        cv::rectangle(vis, cr, cv::Scalar(255, 0, 0), 1);
      }
      const char *status = (last_ellipse_valid_ || last_circle_valid_)
                               ? "landmark: detected"
                               : "landmark: not detected";
      cv::putText(vis, status, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(0, 255, 255), 2);
    }

    cv_bridge::CvImage out_img;
    out_img.header = header;
    out_img.encoding = use_color_output_ ? sensor_msgs::image_encodings::BGR8
                                         : sensor_msgs::image_encodings::MONO8;
    out_img.image = vis;
    image_pub_->publish(*out_img.toImageMsg());
  }

  void publish_lines_data(const std::vector<cv::Vec4i> &segments) {
    std_msgs::msg::Float32MultiArray lines_msg;
    lines_msg.layout.dim.resize(1);
    lines_msg.layout.dim[0].label = "lines_flat_xyxy";
    lines_msg.layout.dim[0].size = segments.size() * 4;
    lines_msg.layout.dim[0].stride = 1;
    lines_msg.data.reserve(segments.size() * 4);
    for (const auto &l : segments) {
      lines_msg.data.push_back(static_cast<float>(l[0]));
      lines_msg.data.push_back(static_cast<float>(l[1]));
      lines_msg.data.push_back(static_cast<float>(l[2]));
      lines_msg.data.push_back(static_cast<float>(l[3]));
    }
    lines_pub_->publish(lines_msg);
  }

  cv::Mat prepare_work_image(const cv::Mat &img, const cv::Rect &roi_rect,
                             double scale) {
    cv::Mat work = img(roi_rect).clone();
    if (downscale_ != 1.0) {
      cv::Mat tmp;
      cv::resize(work, tmp, cv::Size(), 1.0 / scale, 1.0 / scale,
                 cv::INTER_AREA);
      work = tmp;
    }
    return work;
  }

  std::vector<cv::Vec4i> apply_temporal_smoothing(
      const std::vector<cv::Vec4i> &segments_full) {
    std::vector<cv::Vec4i> segments_out;
    if (enable_temporal_smoothing_) {
      segments_out = update_tracks_and_build_output(segments_full);
      if (segments_out.empty() && !segments_full.empty()) {
        // Fallback to raw detections if no stable tracks yet
        segments_out = segments_full;
      }
    } else {
      segments_out = segments_full;
    }
    return segments_out;
  }

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
    // Temporal smoothing clamps
    if (ema_alpha_ < 0.0) ema_alpha_ = 0.0;
    if (ema_alpha_ > 1.0) ema_alpha_ = 1.0;
    if (match_max_px_ < 0) match_max_px_ = 0;
    if (match_max_angle_deg_ < 0.0) match_max_angle_deg_ = 0.0;
    if (match_max_angle_deg_ > 180.0) match_max_angle_deg_ = 180.0;
    if (min_age_to_publish_ < 0) min_age_to_publish_ = 0;
    if (max_missed_ < 0) max_missed_ = 0;
    // Edge closing morphology params
    if (edge_close_kernel_ < 1) edge_close_kernel_ = 1;
    if ((edge_close_kernel_ % 2) == 0) edge_close_kernel_ += 1;  // make odd
    if (edge_close_iter_ < 0) edge_close_iter_ = 0;
    // Calibration thresholds
    if (calib_hsv_s_max_ < 0) calib_hsv_s_max_ = 0;
    if (calib_hsv_s_max_ > 255) calib_hsv_s_max_ = 255;
    if (calib_hsv_v_min_ < 0) calib_hsv_v_min_ = 0;
    if (calib_hsv_v_min_ > 255) calib_hsv_v_min_ = 255;
    if (calib_hsv_v_max_ < 0) calib_hsv_v_max_ = 0;
    if (calib_hsv_v_max_ > 255) calib_hsv_v_max_ = 255;
    if (calib_hsv_v_min_ > calib_hsv_v_max_)
      std::swap(calib_hsv_v_min_, calib_hsv_v_max_);
    if (calib_min_area_ < 0) calib_min_area_ = 0;
    if (calib_min_major_px_ < 0) calib_min_major_px_ = 0;
    if (calib_max_major_ratio_ <= 0.0) calib_max_major_ratio_ = 0.65;
    if (calib_max_major_ratio_ > 1.0) calib_max_major_ratio_ = 1.0;
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
        else if (name == "publish_image_with_lines") {
          // Requires (re)creating publisher. Not supported at runtime.
          result.successful = false;
          result.reason =
              "Changing publish_image_with_lines at runtime is not supported";
        }
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
        // Temporal smoothing params
        else if (name == "enable_temporal_smoothing")
          enable_temporal_smoothing_ = p.as_bool();
        else if (name == "ema_alpha")
          ema_alpha_ = p.as_double();
        else if (name == "match_max_px")
          match_max_px_ = p.as_int();
        else if (name == "match_max_angle_deg")
          match_max_angle_deg_ = p.as_double();
        else if (name == "min_age_to_publish")
          min_age_to_publish_ = p.as_int();
        else if (name == "max_missed")
          max_missed_ = p.as_int();
        // Edge closing params
        else if (name == "use_edge_close")
          use_edge_close_ = p.as_bool();
        else if (name == "edge_close_kernel")
          edge_close_kernel_ = p.as_int();
        else if (name == "edge_close_iter")
          edge_close_iter_ = p.as_int();
        else if (name == "image_topic") {
          // Do not allow changing topic at runtime as it requires
          // resubscription
          result.successful = false;
          result.reason = "Changing image_topic at runtime is not supported";
        } else if (name == "camera_info_topic") {
          result.successful = false;
          result.reason =
              "Changing camera_info_topic at runtime is not supported";
        }
        // Calibration params
        else if (name == "camera_height_meters")
          camera_height_m_ = p.as_double();
        else if (name == "landmark_distance_meters")
          landmark_distance_m_ = p.as_double();
        else if (name == "calib_timeout_sec")
          calib_timeout_sec_ = p.as_double();
        else if (name == "calib_roi")
          calib_roi_ = p.as_integer_array();
        else if (name == "calib_hsv_s_max")
          calib_hsv_s_max_ = p.as_int();
        else if (name == "calib_hsv_v_min")
          calib_hsv_v_min_ = p.as_int();
        else if (name == "calib_hsv_v_max")
          calib_hsv_v_max_ = p.as_int();
        else if (name == "calib_min_area")
          calib_min_area_ = p.as_int();
        else if (name == "calib_min_major_px")
          calib_min_major_px_ = p.as_int();
        else if (name == "calib_max_major_ratio")
          calib_max_major_ratio_ = p.as_double();
        else if (name == "calib_fill_min")
          calib_fill_min_ = p.as_double();
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

    // Initialize calibration timer on first frame
    if (!calib_started_) {
      calib_started_ = true;
      calib_start_time_ = this->now();
    }

    // Step 1: Preprocess image
    cv::Mat img;
    cv::Rect roi_rect;
    double scale;
    cv::Mat gray = preprocess_image(msg, img, roi_rect, scale);
    if (gray.empty()) return;

    // Step 2: Prepare work image for edge detection
    cv::Mat work = prepare_work_image(img, roi_rect, scale);

    // Step 3: Detect edges
    cv::Mat edges = detect_edges(gray, work);

    // Step 4: Detect lines
    std::vector<cv::Vec4i> segments = detect_lines(edges);

    // Step 5: Restore coordinates to original scale and ROI
    std::vector<cv::Vec4i> segments_full =
        restore_coordinates(segments, roi_rect, scale);

    // Step 6: Apply temporal smoothing
    std::vector<cv::Vec4i> segments_out =
        apply_temporal_smoothing_stateful(segments_full);

    // Step 7: Publish results
    std_msgs::msg::Header header = msg->header;
    publish_visualization(segments_out, img, header);
    if (state_ == State::Ready) {
      publish_lines_data(segments_out);
    }
    publish_markers(segments_out, header);

    // Calibration mode processing (detect gray circle and estimate pitch)
    if (state_ == State::CalibratePitch) {
      double v_full = 0.0;
      double x_full = 0.0;
      // reset last detections each frame
      last_circle_valid_ = false;
      last_ellipse_valid_ = false;
      // Optional calibration ROI intersection (full-res coordinates)
      cv::Rect calib_full = valid_roi(img, calib_roi_);
      cv::Rect inter_full = calib_full & roi_rect;
      if (calib_roi_[0] < 0 || calib_roi_[1] < 0 || calib_roi_[2] <= 0 ||
          calib_roi_[3] <= 0) {
        inter_full = roi_rect;
      }
      // Map inter_full to 'work' coordinates
      cv::Rect inter_work;
      inter_work.x = static_cast<int>(
          std::floor((inter_full.x - roi_rect.x) / std::max(1e-6, scale)));
      inter_work.y = static_cast<int>(
          std::floor((inter_full.y - roi_rect.y) / std::max(1e-6, scale)));
      inter_work.width =
          static_cast<int>(std::ceil(inter_full.width / std::max(1e-6, scale)));
      inter_work.height = static_cast<int>(
          std::ceil(inter_full.height / std::max(1e-6, scale)));
      inter_work &= cv::Rect(0, 0, work.cols, work.rows);

      cv::Mat work_sub = work;
      cv::Rect roi_full_for_mapping = roi_rect;
      if (inter_work.width > 0 && inter_work.height > 0) {
        work_sub = work(inter_work).clone();
        roi_full_for_mapping = inter_full;
      }

      if (detect_landmark_center(work_sub, roi_full_for_mapping, scale, x_full,
                                 v_full)) {
        last_circle_px_ = cv::Point2d(x_full, v_full);
        last_circle_valid_ = true;
        v_samples_.push_back(v_full);
        if (v_samples_.size() > kMaxCalibSamples) {
          v_samples_.erase(v_samples_.begin());
        }
      }
      try_finalize_calibration();
    }

    // Step 8: Log timing
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0)
            .count();
    RCLCPP_INFO(this->get_logger(), "Processed frame: %zu lines in %.2f ms",
                segments_out.size(), ms);
  }

  // State-dependent smoothing: force on during calibration, off if disabled
  // later
  std::vector<cv::Vec4i> apply_temporal_smoothing_stateful(
      const std::vector<cv::Vec4i> &segments_full) {
    const bool smoothing =
        (state_ == State::CalibratePitch) ? true : enable_temporal_smoothing_;
    std::vector<cv::Vec4i> segments_out;
    if (smoothing) {
      segments_out = update_tracks_and_build_output(segments_full);
      if (segments_out.empty() && !segments_full.empty()) {
        segments_out = segments_full;
      }
    } else {
      segments_out = segments_full;
    }
    return segments_out;
  }

  // ===== Temporal smoothing implementation =====
  struct Track {
    int id;
    cv::Point2f p1;
    cv::Point2f p2;
    double angle_rad;
    int age;
    int missed;
  };

  // Candidates for calibration landmark (for debug drawing)
  struct CalibCand {
    cv::RotatedRect ellipse;
    double score;
    double ratio;
    double angle_deg;
    double mean_s;
    double mean_v;
    double fill;
  };

  static inline double line_angle_rad(const cv::Point2f &a,
                                      const cv::Point2f &b) {
    return std::atan2(b.y - a.y, b.x - a.x);
  }

  static inline double angle_diff_deg(double a_rad, double b_rad) {
    double d = std::fabs(a_rad - b_rad);
    while (d > CV_PI) d = std::fabs(d - 2 * CV_PI);
    return d * 180.0 / CV_PI;
  }

  static inline double endpoints_min_cost(const cv::Point2f &t1,
                                          const cv::Point2f &t2,
                                          const cv::Point2f &d1,
                                          const cv::Point2f &d2,
                                          bool &swap_det) {
    double c1 = cv::norm(t1 - d1) + cv::norm(t2 - d2);
    double c2 = cv::norm(t1 - d2) + cv::norm(t2 - d1);
    if (c2 < c1) {
      swap_det = true;
      return c2;
    }
    swap_det = false;
    return c1;
  }

  std::vector<cv::Vec4i> update_tracks_and_build_output(
      const std::vector<cv::Vec4i> &detections) {
    // Build detection list
    struct Det {
      cv::Point2f p1;
      cv::Point2f p2;
      double angle_rad;
    };
    std::vector<Det> dets;
    dets.reserve(detections.size());
    for (const auto &v : detections) {
      Det d{cv::Point2f(static_cast<float>(v[0]), static_cast<float>(v[1])),
            cv::Point2f(static_cast<float>(v[2]), static_cast<float>(v[3])),
            0.0};
      d.angle_rad = line_angle_rad(d.p1, d.p2);
      dets.push_back(d);
    }

    // Build candidate matches within thresholds
    struct Cand {
      int t_idx;
      int d_idx;
      double dist_cost;
      bool swap_det;
    };
    const int n_tracks = static_cast<int>(tracks_.size());
    std::vector<Cand> cands;
    cands.reserve(static_cast<size_t>(n_tracks) * dets.size());
    for (int ti = 0; ti < n_tracks; ++ti) {
      const auto &t = tracks_[ti];
      for (int di = 0; di < static_cast<int>(dets.size()); ++di) {
        const auto &d = dets[di];
        double ang_diff = angle_diff_deg(t.angle_rad, d.angle_rad);
        if (ang_diff > match_max_angle_deg_) continue;
        bool swap_det = false;
        double cost = endpoints_min_cost(t.p1, t.p2, d.p1, d.p2, swap_det);
        if (cost <= static_cast<double>(match_max_px_)) {
          cands.push_back(Cand{ti, di, cost, swap_det});
        }
      }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
      return a.dist_cost < b.dist_cost;
    });

    std::vector<int> det_assigned(dets.size(), -1);
    std::vector<int> trk_assigned(n_tracks, -1);

    // Greedy assignment
    for (const auto &c : cands) {
      if (c.t_idx < 0 || c.t_idx >= n_tracks) continue;
      if (c.d_idx < 0 || c.d_idx >= static_cast<int>(dets.size())) continue;
      if (trk_assigned[c.t_idx] != -1) continue;
      if (det_assigned[c.d_idx] != -1) continue;
      trk_assigned[c.t_idx] = c.d_idx;
      det_assigned[c.d_idx] = c.t_idx;
      // Apply EMA update
      auto &t = tracks_[c.t_idx];
      cv::Point2f dp1 = dets[c.d_idx].p1;
      cv::Point2f dp2 = dets[c.d_idx].p2;
      if (c.swap_det) std::swap(dp1, dp2);
      t.p1 = dp1 * static_cast<float>(ema_alpha_) +
             t.p1 * static_cast<float>(1.0 - ema_alpha_);
      t.p2 = dp2 * static_cast<float>(ema_alpha_) +
             t.p2 * static_cast<float>(1.0 - ema_alpha_);
      t.angle_rad = line_angle_rad(t.p1, t.p2);
      t.age += 1;
      t.missed = 0;
    }

    // New tracks for unmatched detections
    for (int di = 0; di < static_cast<int>(dets.size()); ++di) {
      if (det_assigned[di] != -1) continue;
      Track t;
      t.id = next_track_id_++;
      t.p1 = dets[di].p1;
      t.p2 = dets[di].p2;
      t.angle_rad = dets[di].angle_rad;
      t.age = 1;
      t.missed = 0;
      tracks_.push_back(t);
    }

    // Age unmatched tracks
    for (int ti = 0; ti < n_tracks; ++ti) {
      if (trk_assigned[ti] == -1) {
        tracks_[ti].missed += 1;
      }
    }
    // Remove stale tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const Track &t) { return t.missed > max_missed_; }),
        tracks_.end());

    // Build output from mature tracks
    std::vector<cv::Vec4i> out;
    out.reserve(tracks_.size());
    for (const auto &t : tracks_) {
      if (t.age >= min_age_to_publish_) {
        out.emplace_back(cv::Vec4i{static_cast<int>(std::lround(t.p1.x)),
                                   static_cast<int>(std::lround(t.p1.y)),
                                   static_cast<int>(std::lround(t.p2.x)),
                                   static_cast<int>(std::lround(t.p2.y))});
      }
    }
    return out;
  }

  // ===== Calibration: camera pitch estimation =====
  enum class State { CalibratePitch, Ready };

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

  bool detect_landmark_center(const cv::Mat &work_img, const cv::Rect &roi,
                              double scale, double &x_full_out,
                              double &v_full_out) {
    if (work_img.empty()) return false;
    calib_cands_.clear();
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

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    auto work_center = cv::Point2f(static_cast<float>(work_bgr.cols) / 2.0f,
                                   static_cast<float>(work_bgr.rows) / 2.0f);
    bool found = false;
    cv::RotatedRect best_ellipse;
    double best_score = std::numeric_limits<double>::infinity();
    const double max_major =
        calib_max_major_ratio_ *
        static_cast<double>(std::min(work_bgr.cols, work_bgr.rows));
    for (const auto &cnt : contours) {
      if (cnt.size() < 5) continue;
      double area = cv::contourArea(cnt);
      if (area < static_cast<double>(calib_min_area_)) continue;
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
              ? static_cast<double>(cv::countNonZero(mask_inside)) /
                    ellipse_area
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
      // Record candidate for visualization (mapped to full image coords)
      cv::RotatedRect e_full(
          cv::Point2f(static_cast<float>(e.center.x * scale + roi.x),
                      static_cast<float>(e.center.y * scale + roi.y)),
          cv::Size2f(static_cast<float>(e.size.width * scale),
                     static_cast<float>(e.size.height * scale)),
          e.angle);
      calib_cands_.push_back(
          CalibCand{e_full, score, ratio, angle_deg, mean_s, mean_v, fill});
      if (score < best_score) {
        best_score = score;
        best_ellipse = e;
        found = true;
      }
    }

    // Edge-based fallback within HSV mask region
    if (!found) {
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
      for (const auto &cnt : ctr2) {
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

    if (!found) return false;

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
      last_fill_ =
          static_cast<double>(cv::countNonZero(mask_inside)) / sel_area;
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

  void try_finalize_calibration() {
    // Timeout check
    if (calib_started_ && calib_timeout_sec_ > 0.0) {
      const rclcpp::Duration elapsed = this->now() - calib_start_time_;
      if (elapsed.seconds() >= calib_timeout_sec_) {
        RCLCPP_WARN(this->get_logger(),
                    "Calibration timeout after %.1fs. Proceed without pitch.",
                    calib_timeout_sec_);
        transition_to_ready(/*pitch_rad=*/0.0);
        return;
      }
    }

    if (!has_cam_info_) return;
    if (v_samples_.size() < kMinCalibSamples) return;

    const double v_med = median(v_samples_);
    // Compute u = (v - cy)/fy
    const double u = (v_med - cy_) / fy_;
    const double D = landmark_distance_m_;
    const double h = camera_height_m_;
    // tan(theta) = (D*u - h) / (D + h*u)
    const double denom = (D + h * u);
    if (std::abs(denom) < 1e-6) return;  // avoid singularities
    const double t = (D * u - h) / denom;
    const double theta = std::atan(t);

    // Sanity clamp to [-45, 45] deg
    const double max_rad = 45.0 * CV_PI / 180.0;
    const double pitch_rad = std::max(-max_rad, std::min(theta, max_rad));
    if (calib_timeout_sec_ == 0.0) {
      // Continuous calibration mode: keep updating pitch and stay in this state
      estimated_pitch_rad_ = pitch_rad;
      return;
    }
    transition_to_ready(pitch_rad);
  }

  void transition_to_ready(double pitch_rad) {
    estimated_pitch_rad_ = pitch_rad;
    state_ = State::Ready;
    // Disable smoothing per requirement after calibration
    enable_temporal_smoothing_ = false;

    RCLCPP_INFO(this->get_logger(),
                "Calibration finished. Estimated pitch: %.2f deg (%.4f rad)",
                estimated_pitch_rad_ * 180.0 / CV_PI, estimated_pitch_rad_);
  }

  // Parameters
  std::string image_topic_;
  std::string camera_info_topic_;
  bool use_color_output_{};
  bool publish_image_{};
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

  // Edge closing parameters
  bool use_edge_close_{};
  int edge_close_kernel_{};
  int edge_close_iter_{};

  // Temporal smoothing
  bool enable_temporal_smoothing_{};
  double ema_alpha_{};
  int match_max_px_{};
  double match_max_angle_deg_{};
  int min_age_to_publish_{};
  int max_missed_{};
  std::vector<Track> tracks_;
  int next_track_id_{};

  // Calibration / camera model
  State state_{State::CalibratePitch};
  bool has_cam_info_{false};
  double fx_{};
  double fy_{};
  double cx_{};
  double cy_{};
  double camera_height_m_{};
  double landmark_distance_m_{};
  double calib_timeout_sec_{};
  std::vector<int64_t> calib_roi_{};  // optional [x,y,w,h] in full image
  int calib_hsv_s_max_{};
  int calib_hsv_v_min_{};
  int calib_hsv_v_max_{};
  int calib_min_area_{};
  int calib_min_major_px_{};
  double calib_max_major_ratio_{};
  double calib_fill_min_{};
  double estimated_pitch_rad_{std::numeric_limits<double>::quiet_NaN()};
  bool calib_started_{false};
  rclcpp::Time calib_start_time_{};
  std::vector<double> v_samples_{};
  static constexpr size_t kMinCalibSamples = 10;
  static constexpr size_t kMaxCalibSamples = 30;
  bool last_circle_valid_{false};
  cv::Point2d last_circle_px_{};
  bool last_ellipse_valid_{false};
  cv::RotatedRect last_ellipse_full_{};
  std::vector<CalibCand> calib_cands_;
  // Selected landmark metrics (for overlay)
  double last_angle_deg_{};
  double last_ratio_{};
  double last_mean_s_{};
  double last_mean_v_{};
  double last_fill_{};

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
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
