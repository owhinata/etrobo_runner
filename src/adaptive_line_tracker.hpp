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
    int valid_contours;                   // Valid contours (area >= threshold)
    std::vector<int> segment_counts;      // Segment count per valid contour
    int best_contour_id = -1;             // ID of highest scoring contour
    double best_contour_score = 0.0;      // Score of highest scoring contour
    std::vector<cv::Point> best_contour;  // Points of highest scoring contour
  };

  explicit AdaptiveLineTracker(LineDetectorNode* node);
  ~AdaptiveLineTracker();

  // Initialize and declare tracking parameters
  void declare_parameters();

  // Try to update a parameter if it belongs to AdaptiveLineTracker
  bool try_update_parameter(const rclcpp::Parameter& param);

  // Main processing method for line detection
  // Returns true if detection was successful
  bool process_frame(const cv::Mat& img, DetectionResult& result);

  // Draw visualization overlay for image_with_lines output
  void draw_visualization_overlay(cv::Mat& img) const;

  // Reset the tracker
  void reset();

  // Get tracking confidence (0.0 to 1.0)
  double get_confidence() const;

  // Configuration parameters
  struct Config {
    double min_line_width = 10.0;  // Minimum expected line width in pixels
    int scan_step = 5;             // Vertical step between scan lines

    // Parameters for width-based scoring
    double width_importance = 2.0;  // Importance of width in scoring (1.0-5.0)
    double min_contour_score = 10.0;  // Minimum score for valid contour
    int min_segments_straight = 5;    // Min segments for straight line
    int min_segments_curve = 3;       // Min segments for curve
  };

  void set_config(const Config& config);
  Config get_config() const;

 private:
  // Pimpl idiom
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

#endif  // ADAPTIVE_LINE_TRACKER_HPP_