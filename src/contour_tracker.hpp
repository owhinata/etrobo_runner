#ifndef CONTOUR_TRACKER_HPP_
#define CONTOUR_TRACKER_HPP_

#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

class ContourTracker {
 public:
  struct TrackedContour {
    int id;                          // Unique ID for this contour
    std::vector<cv::Point> contour;  // Current contour points
    cv::Point2f centroid;            // Centroid position
    cv::Point2f predicted_centroid;  // Kalman predicted position
    cv::Rect bounding_box;           // Bounding rectangle
    double area;                     // Contour area
    int age;                         // Number of frames tracked
    int missed_frames;               // Consecutive frames not detected
    cv::KalmanFilter kf;             // Kalman filter for this contour

    TrackedContour();
    void init_kalman_filter();
    void predict();
    void update(const cv::Point2f& measured_centroid);
  };

  ContourTracker();
  ~ContourTracker() = default;

  // Update tracking with new contours from current frame
  void update(const std::vector<std::vector<cv::Point>>& contours);

  // Get currently tracked contours
  const std::map<int, TrackedContour>& get_tracked_contours() const {
    return tracked_contours_;
  }

  // Get stable contours (tracked for minimum frames)
  std::vector<TrackedContour> get_stable_contours(int min_age = 3) const;

  // Configuration
  void set_max_missed_frames(int frames) { max_missed_frames_ = frames; }
  void set_max_distance_threshold(double distance) {
    max_distance_threshold_ = distance;
  }
  void set_min_contour_area(double area) { min_contour_area_ = area; }

 private:
  // Calculate centroid of contour
  cv::Point2f calculate_centroid(const std::vector<cv::Point>& contour);

  // Find best match for a tracked contour in new contours
  int find_best_match(const TrackedContour& tracked,
                      const std::vector<std::vector<cv::Point>>& new_contours,
                      const std::vector<cv::Point2f>& new_centroids,
                      std::vector<bool>& used_indices);

  // Calculate distance between two points
  double calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2);

  // Calculate IoU between two bounding boxes
  double calculate_iou(const cv::Rect& r1, const cv::Rect& r2);

  // Tracked contours map (ID -> TrackedContour)
  std::map<int, TrackedContour> tracked_contours_;

  // Next available ID for new contours
  int next_id_;

  // Tracking parameters
  int max_missed_frames_;          // Remove after this many missed frames
  double max_distance_threshold_;  // Maximum distance for matching (pixels)
  double min_contour_area_;        // Minimum area to track
};

#endif  // CONTOUR_TRACKER_HPP_