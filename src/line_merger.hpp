#ifndef LINE_MERGER_HPP_
#define LINE_MERGER_HPP_

#include <map>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

#include "src/contour_tracker.hpp"

class LineMerger {
 public:
  enum class MergeMethod {
    DIRECTION_ENDPOINT,  // Method 1+2: Direction vector + endpoint distance
    KALMAN_GRAPH         // Method 4+5: Kalman prediction + graph-based
  };

  struct MergeConfig {
    // Common parameters
    MergeMethod method = MergeMethod::DIRECTION_ENDPOINT;
    bool enabled = false;

    // Direction + Endpoint method parameters
    float max_angle_diff = 20.0f;     // Maximum angle difference in degrees
    float max_endpoint_dist = 50.0f;  // Maximum endpoint distance in pixels
    float min_line_length = 30.0f;    // Minimum line length to consider

    // Kalman + Graph method parameters
    int prediction_frames = 5;  // Frames to predict ahead
    float trajectory_threshold =
        30.0f;                      // Threshold for trajectory intersection
    float merge_confidence = 0.7f;  // Minimum confidence for merging
  };

  struct LineSegment {
    int id;
    std::vector<cv::Point> points;
    cv::Point2f start_point;
    cv::Point2f end_point;
    cv::Vec2f direction;
    double length;

    LineSegment() : id(-1), length(0) {}
  };

  struct MergeCandidate {
    int line1_id;
    int line2_id;
    float score;  // Higher score = better merge candidate

    bool operator<(const MergeCandidate& other) const {
      return score > other.score;  // Sort by descending score
    }
  };

  LineMerger();

  // Configuration
  void set_config(const MergeConfig& config) { config_ = config; }
  MergeConfig get_config() const { return config_; }

  // Main merging function
  std::vector<std::vector<cv::Point>> merge_lines(
      const std::map<int, ContourTracker::TrackedContour>& tracked_contours);

  // Get merge groups (for visualization)
  std::map<int, std::set<int>> get_merge_groups() const {
    return merge_groups_;
  }

 private:
  MergeConfig config_;
  std::map<int, std::set<int>> merge_groups_;  // Groups of merged line IDs

  // Method 1+2: Direction and Endpoint based merging
  std::vector<MergeCandidate> find_merge_candidates_direction(
      const std::map<int, LineSegment>& line_segments);
  cv::Vec2f calculate_line_direction(const std::vector<cv::Point>& points);
  void extract_line_endpoints(LineSegment& segment);
  float calculate_angle_difference(const cv::Vec2f& dir1,
                                   const cv::Vec2f& dir2);
  float calculate_endpoint_distance(const LineSegment& seg1,
                                    const LineSegment& seg2);

  // Method 4+5: Kalman and Graph based merging
  std::vector<MergeCandidate> find_merge_candidates_kalman(
      const std::map<int, ContourTracker::TrackedContour>& tracked_contours);
  cv::Point2f predict_future_position(
      const ContourTracker::TrackedContour& contour, int frames_ahead);
  bool check_trajectory_intersection(
      const ContourTracker::TrackedContour& contour1,
      const ContourTracker::TrackedContour& contour2, int frames_ahead);

  // Graph operations
  void build_merge_graph(const std::vector<MergeCandidate>& candidates);
  void find_connected_components();

  // Merging operations
  std::vector<cv::Point> merge_point_sets(
      const std::vector<std::vector<cv::Point>>& point_sets);
  void sort_points_along_line(std::vector<cv::Point>& points);
};

#endif  // LINE_MERGER_HPP_