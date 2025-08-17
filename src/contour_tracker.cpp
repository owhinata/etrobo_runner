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

  // Process noise covariance - will be set from ContourTracker
  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));  // Default

  // Measurement noise covariance - will be set from ContourTracker
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(5e-2));  // Default

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
      max_distance_threshold_(75.0),  // Increased for better curve tracking
      min_contour_area_(20.0) {}

void ContourTracker::update(
    const std::vector<std::vector<cv::Point>>& contours) {
  // If tracking is disabled, clear tracked contours and return
  if (!enabled_) {
    tracked_contours_.clear();
    return;
  }

  // Debug: Frame summary
  if (debug_enabled_) {
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 10 == 0) {  // Every 10 frames
      printf("\n[Tracker] === Frame %d Summary ===\n", frame_count);
      printf("[Tracker] Tracked: %zu, New contours: %zu\n",
             tracked_contours_.size(), contours.size());

      // Log each tracked contour's state
      for (const auto& [id, tracked] : tracked_contours_) {
        float vx = tracked.kf.statePost.at<float>(2);
        float vy = tracked.kf.statePost.at<float>(3);
        float speed = std::sqrt(vx * vx + vy * vy);
        printf(
            "  ID%d: pos(%.0f,%.0f) vel(%.1f,%.1f) speed=%.1f age=%d miss=%d\n",
            id, tracked.centroid.x, tracked.centroid.y, vx, vy, speed,
            tracked.age, tracked.missed_frames);
      }
    }
  }

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

  new_centroids.reserve(contours.size());
  new_bboxes.reserve(contours.size());
  new_areas.reserve(contours.size());

  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area < min_contour_area_) {
      new_centroids.push_back(cv::Point2f(-1, -1));  // Invalid
      new_bboxes.push_back(cv::Rect());
      new_areas.push_back(0);
      used_indices[i] = true;  // Skip this contour
    } else {
      new_centroids.push_back(calculate_centroid(contours[i]));
      new_bboxes.push_back(cv::boundingRect(contours[i]));
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

      // Debug: Log successful match
      if (debug_enabled_ && id < 5) {
        double match_distance = calculate_distance(tracked.predicted_centroid,
                                                   new_centroids[best_match]);
        float prediction_error =
            calculate_distance(tracked.centroid, new_centroids[best_match]);
        printf(
            "[Tracker] ID%d matched: idx=%d, match_dist=%.1f, pred_error=%.1f, "
            "area=%.1f\n",
            id, best_match, match_distance, prediction_error,
            new_areas[best_match]);
      }
    } else {
      // No match found - use prediction
      tracked.centroid = tracked.predicted_centroid;

      // Debug: Log miss
      if (debug_enabled_ && id < 5) {
        printf(
            "[Tracker] ID%d MISSED: using predicted(%.1f,%.1f), "
            "miss_count=%d\n",
            id, tracked.predicted_centroid.x, tracked.predicted_centroid.y,
            tracked.missed_frames);
      }

      // Mark for removal if missed too many frames
      if (tracked.missed_frames > max_missed_frames_) {
        to_remove.push_back(id);
        if (debug_enabled_ && id < 5) {
          printf("[Tracker] ID%d REMOVED: exceeded max_missed_frames=%d\n", id,
                 max_missed_frames_);
        }
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

      // Set Kalman filter parameters from tracker settings
      cv::setIdentity(new_tracked.kf.processNoiseCov,
                      cv::Scalar::all(process_noise_));
      cv::setIdentity(new_tracked.kf.measurementNoiseCov,
                      cv::Scalar::all(measurement_noise_));

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

  // Adaptive threshold based on velocity
  float vx = tracked.kf.statePost.at<float>(2);
  float vy = tracked.kf.statePost.at<float>(3);
  float speed = std::sqrt(vx * vx + vy * vy);

  // Increase threshold for fast-moving objects
  double adaptive_threshold = max_distance_threshold_;
  if (speed > speed_threshold_) {                        // If moving fast
    adaptive_threshold = max_distance_threshold_ * 1.5;  // 50% more tolerance
  }

  double best_distance = adaptive_threshold;

  // Debug: Log search start
  if (debug_enabled_ && tracked.id < 5) {
    printf(
        "[Tracker] ID%d finding match: pred(%.1f,%.1f), speed=%.1f, "
        "thresh=%.1f, candidates=%zu\n",
        tracked.id, tracked.predicted_centroid.x, tracked.predicted_centroid.y,
        speed, adaptive_threshold, new_contours.size());
  }

  for (size_t i = 0; i < new_contours.size(); i++) {
    if (used_indices[i]) continue;

    // Skip invalid centroids
    if (new_centroids[i].x < 0 || new_centroids[i].y < 0) continue;

    // Use predicted position for matching
    double distance =
        calculate_distance(tracked.predicted_centroid, new_centroids[i]);

    // Debug: Log each candidate
    if (debug_enabled_ && tracked.id < 5 &&
        distance < max_distance_threshold_ * 2) {
      printf(
          "  [Tracker] ID%d -> candidate[%zu]: pos(%.1f,%.1f) dist=%.1f %s\n",
          tracked.id, i, new_centroids[i].x, new_centroids[i].y, distance,
          distance < best_distance ? "(better)" : "");
    }

    if (distance < best_distance) {
      best_distance = distance;
      best_idx = i;
    }
  }

  // Debug: Log result
  if (debug_enabled_ && tracked.id < 5) {
    if (best_idx >= 0) {
      printf("[Tracker] ID%d best match: idx=%d, dist=%.1f (thresh=%.1f)\n",
             tracked.id, best_idx, best_distance, adaptive_threshold);
    } else {
      printf("[Tracker] ID%d NO MATCH: best_dist=%.1f > thresh=%.1f\n",
             tracked.id, best_distance, adaptive_threshold);
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