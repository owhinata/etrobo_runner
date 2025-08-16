#ifndef ADAPTIVE_LINE_TRACKER_HPP_
#define ADAPTIVE_LINE_TRACKER_HPP_

#include <memory>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>

// Forward declaration
class LineDetectorNode;

class AdaptiveLineTracker {
 public:
  // Structure to hold tracked line information for each contour
  struct TrackedLine {
    int contour_id;                   // Contour index
    std::vector<cv::Point2d> points;  // Segment points for this contour
    double area;                      // Contour area

    TrackedLine(int id = -1, double a = 0.0) : contour_id(id), area(a) {}
  };

  // Structure to hold detection results and statistics
  struct DetectionResult {
    std::vector<TrackedLine> tracked_lines;  // Detected lines
    int total_scans;                         // Total scan lines processed
    int successful_detections;               // Scans with detected segments
    int total_contours;                      // Total contours found
    int valid_contours;               // Valid contours (area >= threshold)
    std::vector<int> segment_counts;  // Segment count per valid contour
  };

  explicit AdaptiveLineTracker(LineDetectorNode* node);
  ~AdaptiveLineTracker();

  // Initialize and declare tracking parameters
  void declare_parameters();

  // Try to update a parameter if it belongs to AdaptiveLineTracker
  bool try_update_parameter(const rclcpp::Parameter& param);

  // Main processing method for line detection
  // Returns true if detection was successful
  bool process_frame(const cv::Mat& img, const cv::Rect& roi,
                     DetectionResult& result);

  // Draw visualization overlay for image_with_lines output
  void draw_visualization_overlay(cv::Mat& img, const DetectionResult& result,
                                  const cv::Rect& roi) const;

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