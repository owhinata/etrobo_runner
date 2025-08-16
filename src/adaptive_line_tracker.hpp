#ifndef ADAPTIVE_LINE_TRACKER_HPP_
#define ADAPTIVE_LINE_TRACKER_HPP_

#include <memory>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>

class AdaptiveLineTracker {
 public:
  // Structure to hold tracked line information for each contour
  struct TrackedLine {
    int contour_id;                   // Contour index
    std::vector<cv::Point2d> points;  // Segment points for this contour
    double area;                      // Contour area

    TrackedLine(int id = -1, double a = 0.0) : contour_id(id), area(a) {}
  };

  explicit AdaptiveLineTracker(rclcpp::Node* node = nullptr);
  ~AdaptiveLineTracker();

  // Main tracking function (returns tracked lines per contour in mask's
  // coordinate system) Optional total_processing_ms parameter to include in log
  // output
  std::vector<TrackedLine> track_line(const cv::Mat& black_mask,
                                      double total_processing_ms = -1.0);

  // Reset the tracker
  void reset();

  // Get tracking confidence (0.0 to 1.0)
  double get_confidence() const;

  // Configuration parameters
  struct Config {
    double max_line_width = 100.0;  // Maximum expected line width in pixels
    double min_line_width = 10.0;   // Minimum expected line width in pixels
    double max_lateral_jump =
        30.0;                      // Maximum allowed lateral jump between scans
    int scan_step = 5;             // Vertical step between scan lines
    double width_weight = 0.2;     // Weight for width change in scoring
    double position_weight = 0.3;  // Weight for position change in scoring
    double prediction_weight = 0.5;  // Weight for prediction error in scoring

    // New parameters for width-based scoring
    double width_importance = 2.0;  // Importance of width in scoring (1.0-5.0)
    double min_contour_score = 10.0;    // Minimum score for valid contour
    double curve_width_threshold = 30;  // Width threshold for curve detection
    int min_segments_straight = 5;      // Min segments for straight line
    int min_segments_curve = 3;         // Min segments for curve
  };

  void set_config(const Config& config);
  Config get_config() const;

 private:
  // Pimpl idiom
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

#endif  // ADAPTIVE_LINE_TRACKER_HPP_