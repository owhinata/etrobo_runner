#ifndef ADAPTIVE_LINE_TRACKER_HPP_
#define ADAPTIVE_LINE_TRACKER_HPP_

#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

class AdaptiveLineTracker {
 public:
  AdaptiveLineTracker();
  ~AdaptiveLineTracker() = default;

  // Main tracking function
  std::vector<cv::Point2d> track_line(const cv::Mat& black_mask,
                                      const cv::Rect& roi);

  // Reset the tracker
  void reset();

  // Get tracking confidence (0.0 to 1.0)
  double get_confidence() const { return confidence_; }

  // Configuration parameters
  struct Config {
    double max_line_width = 100.0;  // Maximum expected line width in pixels
    double min_line_width = 10.0;   // Minimum expected line width in pixels
    double max_lateral_jump =
        30.0;                      // Maximum allowed lateral jump between scans
    int scan_step = 5;             // Vertical step between scan lines
    int smooth_window = 3;         // Window size for trajectory smoothing
    double width_weight = 0.2;     // Weight for width change in scoring
    double position_weight = 0.3;  // Weight for position change in scoring
    double prediction_weight = 0.5;  // Weight for prediction error in scoring
  };

  void set_config(const Config& config) { config_ = config; }
  Config get_config() const { return config_; }

 private:
  // Segment structure for black regions
  struct Segment {
    int start_x;
    int end_x;

    double center() const { return (start_x + end_x) / 2.0; }
    double width() const { return end_x - start_x; }
  };

  // Find black segments in a horizontal scan line
  std::vector<Segment> find_black_segments(const cv::Mat& mask, int y,
                                           const cv::Rect& roi);

  // Select the best segment based on previous position and Kalman prediction
  int select_best_segment(const std::vector<Segment>& segments,
                          const cv::Point2d& prev_center, double prev_width,
                          int y);

  // Initialize Kalman filter with first detection
  void initialize_kalman(const cv::Point2d& point, int y);

  // Check if tracking should be reset based on deviation
  bool should_reset_tracking(double measured_x, double predicted_x);

  // Smooth the tracked trajectory
  std::vector<cv::Point2d> smooth_trajectory(
      const std::vector<cv::Point2d>& points);

  // Update confidence based on tracking quality
  void update_confidence(bool detection_success, double prediction_error);

  // Member variables
  cv::KalmanFilter kf_;
  bool kf_initialized_;
  std::vector<cv::Point2d> tracked_points_;
  double confidence_;
  int consecutive_failures_;
  int deviation_count_;
  Config config_;

  // Previous frame data for continuity
  cv::Point2d prev_center_;
  double prev_width_;
};

#endif  // ADAPTIVE_LINE_TRACKER_HPP_