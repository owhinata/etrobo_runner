#ifndef ADAPTIVE_LINE_TRACKER_HPP_
#define ADAPTIVE_LINE_TRACKER_HPP_

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

class AdaptiveLineTracker {
 public:
  AdaptiveLineTracker();
  ~AdaptiveLineTracker();

  // Main tracking function (returns points in mask's coordinate system)
  std::vector<cv::Point2d> track_line(const cv::Mat& black_mask);

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
  };

  void set_config(const Config& config);
  Config get_config() const;

 private:
  // Pimpl idiom
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

#endif  // ADAPTIVE_LINE_TRACKER_HPP_