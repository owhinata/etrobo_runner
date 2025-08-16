#include "src/contour_tracker.hpp"

#include <algorithm>
#include <cmath>

// TrackedContour implementation
ContourTracker::TrackedContour::TrackedContour()
    : id(-1), age(0), missed_frames(0) {
  init_kalman_filter();
}

void ContourTracker::TrackedContour::init_kalman_filter() {
  // State: [x, y, vx, vy] (position and velocity)
  // Measurement: [x, y] (position only)
  kf.init(4, 2, 0);

  // State transition matrix (constant velocity model)
  // [1, 0, dt, 0 ]
  // [0, 1, 0,  dt]
  // [0, 0, 1,  0 ]
  // [0, 0, 0,  1 ]
  float dt = 1.0f;  // Assuming frame-to-frame
  kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0,
                         0, 1, 0, 0, 0, 0, 1);

  // Measurement matrix
  // [1, 0, 0, 0]
  // [0, 1, 0, 0]
  kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

  // Process noise covariance
  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));

  // Measurement noise covariance
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));

  // Error covariance
  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

void ContourTracker::TrackedContour::predict() {
  // For first frame after initialization, use current centroid
  if (age <= 1) {
    predicted_centroid = centroid;  // Use last known position
    return;
  }

  cv::Mat prediction = kf.predict();
  predicted_centroid.x = prediction.at<float>(0);
  predicted_centroid.y = prediction.at<float>(1);
}

void ContourTracker::TrackedContour::update(
    const cv::Point2f& measured_centroid) {
  // Initialize Kalman filter state if first measurement
  if (age == 0) {
    kf.statePre.at<float>(0) = measured_centroid.x;
    kf.statePre.at<float>(1) = measured_centroid.y;
    kf.statePre.at<float>(2) = 0;
    kf.statePre.at<float>(3) = 0;

    kf.statePost.at<float>(0) = measured_centroid.x;
    kf.statePost.at<float>(1) = measured_centroid.y;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;
  }

  // Create measurement
  cv::Mat measurement =
      (cv::Mat_<float>(2, 1) << measured_centroid.x, measured_centroid.y);

  // Correct with measurement
  cv::Mat corrected = kf.correct(measurement);
  centroid.x = corrected.at<float>(0);
  centroid.y = corrected.at<float>(1);
}

// ContourTracker implementation
ContourTracker::ContourTracker()
    : next_id_(0),
      max_missed_frames_(5),
      max_distance_threshold_(50.0),
      min_contour_area_(20.0) {}

void ContourTracker::update(
    const std::vector<std::vector<cv::Point>>& contours) {
  // Step 1: Predict positions for all tracked contours
  for (auto& [id, tracked] : tracked_contours_) {
    tracked.predict();
    tracked.missed_frames++;  // Will be reset if matched
  }

  // Step 2: Calculate centroids and properties for new contours
  std::vector<cv::Point2f> new_centroids;
  std::vector<cv::Rect> new_bboxes;
  std::vector<double> new_areas;
  std::vector<bool> used_indices(contours.size(), false);

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area < min_contour_area_) {
      new_centroids.push_back(cv::Point2f(-1, -1));  // Invalid
      new_bboxes.push_back(cv::Rect());
      new_areas.push_back(0);
      used_indices[new_centroids.size() - 1] = true;  // Skip this contour
    } else {
      new_centroids.push_back(calculate_centroid(contour));
      new_bboxes.push_back(cv::boundingRect(contour));
      new_areas.push_back(area);
    }
  }

  // Step 3: Match existing tracked contours with new contours
  std::vector<int> to_remove;

  for (auto& [id, tracked] : tracked_contours_) {
    int best_match =
        find_best_match(tracked, contours, new_centroids, used_indices);

    if (best_match >= 0) {
      // Update tracked contour with new detection
      tracked.contour = contours[best_match];
      tracked.bounding_box = new_bboxes[best_match];
      tracked.area = new_areas[best_match];
      tracked.update(new_centroids[best_match]);
      tracked.age++;
      tracked.missed_frames = 0;
      used_indices[best_match] = true;
    } else {
      // No match found - use prediction
      tracked.centroid = tracked.predicted_centroid;

      // Mark for removal if missed too many frames
      if (tracked.missed_frames > max_missed_frames_) {
        to_remove.push_back(id);
      }
    }
  }

  // Step 4: Remove lost contours
  for (int id : to_remove) {
    tracked_contours_.erase(id);
  }

  // Step 5: Add new contours that weren't matched
  for (size_t i = 0; i < contours.size(); i++) {
    if (!used_indices[i] && new_areas[i] >= min_contour_area_) {
      TrackedContour new_tracked;
      new_tracked.id = next_id_++;
      new_tracked.contour = contours[i];
      new_tracked.centroid = new_centroids[i];
      new_tracked.predicted_centroid = new_centroids[i];
      new_tracked.bounding_box = new_bboxes[i];
      new_tracked.area = new_areas[i];
      new_tracked.age =
          0;  // Start with 0 so update() initializes Kalman filter
      new_tracked.missed_frames = 0;
      new_tracked.update(new_centroids[i]);
      new_tracked.age = 1;  // Then set to 1 after initialization

      tracked_contours_[new_tracked.id] = new_tracked;
    }
  }
}

std::vector<ContourTracker::TrackedContour> ContourTracker::get_stable_contours(
    int min_age) const {
  std::vector<TrackedContour> stable;
  for (const auto& [id, tracked] : tracked_contours_) {
    if (tracked.age >= min_age && tracked.missed_frames == 0) {
      stable.push_back(tracked);
    }
  }
  return stable;
}

cv::Point2f ContourTracker::calculate_centroid(
    const std::vector<cv::Point>& contour) {
  cv::Moments m = cv::moments(contour);
  if (m.m00 == 0) return cv::Point2f(0, 0);
  return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
}

int ContourTracker::find_best_match(
    const TrackedContour& tracked,
    const std::vector<std::vector<cv::Point>>& new_contours,
    const std::vector<cv::Point2f>& new_centroids,
    std::vector<bool>& used_indices) {
  int best_idx = -1;
  double best_distance = max_distance_threshold_;

  for (size_t i = 0; i < new_contours.size(); i++) {
    if (used_indices[i]) continue;

    // Skip invalid centroids
    if (new_centroids[i].x < 0 || new_centroids[i].y < 0) continue;

    // Use predicted position for matching
    double distance =
        calculate_distance(tracked.predicted_centroid, new_centroids[i]);

    if (distance < best_distance) {
      best_distance = distance;
      best_idx = i;
    }
  }

  return best_idx;
}

double ContourTracker::calculate_distance(const cv::Point2f& p1,
                                          const cv::Point2f& p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt(dx * dx + dy * dy);
}

double ContourTracker::calculate_iou(const cv::Rect& r1, const cv::Rect& r2) {
  cv::Rect intersection = r1 & r2;
  if (intersection.area() == 0) return 0.0;

  double union_area = r1.area() + r2.area() - intersection.area();
  return static_cast<double>(intersection.area()) / union_area;
}