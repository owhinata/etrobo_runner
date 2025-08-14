#include "src/adaptive_line_tracker.hpp"

#include <algorithm>
#include <cmath>

AdaptiveLineTracker::AdaptiveLineTracker()
    : kf_initialized_(false),
      confidence_(0.0),
      consecutive_failures_(0),
      deviation_count_(0),
      prev_width_(30.0) {
  // Initialize Kalman Filter
  // State: [x, dx/dy] (x position and slope at each scan line)
  kf_ = cv::KalmanFilter(2, 1, 0);

  // State transition matrix (constant velocity model)
  // x' = x + dx/dy * Δy
  // dx/dy' = dx/dy
  kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1,  // x position update
                          0, 1);  // slope remains constant

  // Measurement matrix (we only measure x position)
  kf_.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);

  // Process noise covariance (reduced for more stable prediction)
  cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-5));

  // Measurement noise covariance (reduced for more trust in measurements)
  cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(5e-2));

  // Initial error covariance
  cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
}

void AdaptiveLineTracker::reset() {
  kf_initialized_ = false;
  tracked_points_.clear();
  confidence_ = 0.0;
  consecutive_failures_ = 0;
  deviation_count_ = 0;
  prev_width_ = 30.0;
}

std::vector<cv::Point2d> AdaptiveLineTracker::track_line(
    const cv::Mat& black_mask, const cv::Rect& roi) {
  tracked_points_.clear();

  if (black_mask.empty() || roi.width <= 0 || roi.height <= 0) {
    return tracked_points_;
  }

  // Initialize previous center to middle of ROI at bottom
  if (!kf_initialized_) {
    // Start from the bottom center, but look for actual black line
    prev_center_ = cv::Point2d(roi.x + roi.width / 2.0, roi.y + roi.height);

    // Find initial black line position at the bottom
    int init_y = roi.y + roi.height - 10;
    auto init_segments = find_black_segments(black_mask, init_y, roi);
    if (!init_segments.empty()) {
      // Find segment closest to center
      double center_x = roi.x + roi.width / 2.0;
      int best_idx = 0;
      double min_dist = std::abs(init_segments[0].center() - center_x);
      for (size_t i = 1; i < init_segments.size(); i++) {
        double dist = std::abs(init_segments[i].center() - center_x);
        if (dist < min_dist) {
          min_dist = dist;
          best_idx = i;
        }
      }
      prev_center_.x = init_segments[best_idx].center();
      prev_width_ = init_segments[best_idx].width();
    }
  }

  std::vector<cv::Point2d> candidate_points;

  // Bottom-up scanning from robot position upward
  int consecutive_missing = 0;
  const int max_missing = 6;  // Allow more missing scans (for gray disk)

  for (int y = roi.y + roi.height - 10; y > roi.y; y -= config_.scan_step) {
    auto segments = find_black_segments(black_mask, y, roi);

    if (segments.empty()) {
      consecutive_missing++;
      // No black region found - use Kalman prediction if available
      if (kf_initialized_) {
        cv::Mat prediction = kf_.predict();
        float pred_x = prediction.at<float>(0);

        // Check if we might be over the gray disk (landmark)
        // Gray disk is typically around y=200-300 in the image
        bool possibly_over_landmark =
            (y > roi.y + roi.height * 0.3 && y < roi.y + roi.height * 0.7);

        if (possibly_over_landmark || consecutive_missing <= max_missing) {
          // Add predicted point
          candidate_points.push_back(cv::Point2d(pred_x, y));
          consecutive_failures_++;

          // Update confidence (less penalty if possibly over landmark)
          update_confidence(false, possibly_over_landmark ? 0.5 : 0.0);
        } else if (consecutive_missing > max_missing * 2) {
          // Allow more missing segments before stopping
          // This helps bridge over the gray disk
          break;
        }
      }
      continue;
    }

    // Reset consecutive missing counter when we find segments
    consecutive_missing = 0;

    // Select best segment
    int best_idx = select_best_segment(segments, prev_center_, prev_width_, y);

    if (best_idx >= 0) {
      double center_x = segments[best_idx].center();
      cv::Point2d current_point(center_x, y);

      if (!kf_initialized_) {
        // First detection - initialize Kalman filter
        initialize_kalman(current_point, y);
        kf_initialized_ = true;
        candidate_points.push_back(current_point);
      } else {
        // Get prediction
        cv::Mat prediction = kf_.predict();
        float pred_x = prediction.at<float>(0);
        double prediction_error = std::abs(center_x - pred_x);

        if (prediction_error < config_.max_lateral_jump) {
          // Good match - update Kalman filter
          cv::Mat measurement = (cv::Mat_<float>(1, 1) << center_x);
          kf_.correct(measurement);
          candidate_points.push_back(current_point);
          consecutive_failures_ = 0;

          // Update confidence
          update_confidence(true, prediction_error);
        } else {
          // Large deviation - check if we should reset
          if (should_reset_tracking(center_x, pred_x)) {
            initialize_kalman(current_point, y);
            candidate_points.push_back(current_point);
          } else {
            // Use prediction instead of measurement
            candidate_points.push_back(cv::Point2d(pred_x, y));
          }
        }
      }

      prev_center_ = current_point;
      prev_width_ = segments[best_idx].width();
    }
  }

  // Apply smoothing if we have enough points
  if (candidate_points.size() > config_.smooth_window) {
    tracked_points_ = smooth_trajectory(candidate_points);
  } else {
    tracked_points_ = candidate_points;
  }

  return tracked_points_;
}

std::vector<AdaptiveLineTracker::Segment>
AdaptiveLineTracker::find_black_segments(const cv::Mat& mask, int y,
                                         const cv::Rect& roi) {
  std::vector<Segment> segments;

  // Check if y is within the mask bounds (mask is ROI-cropped)
  int y_in_mask = y - roi.y;
  if (y_in_mask < 0 || y_in_mask >= mask.rows) {
    return segments;
  }

  bool in_black = false;
  int start_x = 0;

  // Note: mask is already ROI-cropped, so use relative coordinates
  for (int x = 0; x < roi.width && x < mask.cols; x++) {
    bool is_black = mask.at<uchar>(y_in_mask, x) > 128;

    if (is_black && !in_black) {
      start_x = x;
      in_black = true;
    } else if (!is_black && in_black) {
      double width = x - start_x;
      if (width >= config_.min_line_width && width <= config_.max_line_width) {
        // Convert back to absolute coordinates
        segments.push_back({start_x + roi.x, x + roi.x});
      }
      in_black = false;
    }
  }

  // Handle segment that extends to edge of ROI
  if (in_black) {
    int end_x = std::min(roi.width, mask.cols);
    double width = end_x - start_x;
    if (width >= config_.min_line_width && width <= config_.max_line_width) {
      // Convert back to absolute coordinates
      segments.push_back({start_x + roi.x, end_x + roi.x});
    }
  }

  return segments;
}

int AdaptiveLineTracker::select_best_segment(
    const std::vector<Segment>& segments, const cv::Point2d& prev_center,
    double prev_width, int y) {
  if (segments.empty()) {
    return -1;
  }

  int best_idx = -1;
  double best_score = std::numeric_limits<double>::max();

  // Add maximum allowed distance from previous center
  const double max_distance = config_.max_lateral_jump * 2.0;

  // Get Kalman prediction if available
  float pred_x = prev_center.x;
  if (kf_initialized_) {
    // Save current state
    cv::Mat state_backup = kf_.statePost.clone();
    cv::Mat cov_backup = kf_.errorCovPost.clone();

    // Get prediction
    cv::Mat prediction = kf_.predict();
    pred_x = prediction.at<float>(0);

    // Restore state (so we can call predict again later)
    kf_.statePost = state_backup;
    kf_.errorCovPost = cov_backup;
  }

  for (size_t i = 0; i < segments.size(); i++) {
    double center = segments[i].center();
    double width = segments[i].width();

    // Calculate errors
    double position_error = std::abs(center - prev_center.x);
    double prediction_error = std::abs(center - pred_x);
    double width_error = std::abs(width - prev_width) / prev_width;

    // Reject segments that are too far from previous position
    if (position_error > max_distance) {
      continue;
    }

    // Reject segments with sudden width changes (likely noise)
    if (width_error > 0.5 && i > 0) {  // Allow 50% width change max
      continue;
    }

    // Weighted score
    double score = config_.position_weight * position_error +
                   config_.prediction_weight * prediction_error +
                   config_.width_weight * width_error * prev_width;

    if (score < best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  return best_idx;
}

void AdaptiveLineTracker::initialize_kalman(const cv::Point2d& point, int y) {
  // Initialize state: [x, dx/dy]
  // Initially assume vertical line (dx/dy = 0)
  kf_.statePost = (cv::Mat_<float>(2, 1) << point.x, 0);

  // Reset error covariance
  cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

  // Update scan step in transition matrix based on config
  kf_.transitionMatrix.at<float>(0, 1) = config_.scan_step;

  deviation_count_ = 0;
  confidence_ = 0.5;  // Start with medium confidence
}

bool AdaptiveLineTracker::should_reset_tracking(double measured_x,
                                                double predicted_x) {
  if (std::abs(measured_x - predicted_x) > config_.max_lateral_jump) {
    deviation_count_++;
    if (deviation_count_ > 3) {
      deviation_count_ = 0;
      return true;
    }
  } else {
    deviation_count_ = std::max(0, deviation_count_ - 1);
  }

  // Also reset if we've had too many consecutive failures
  if (consecutive_failures_ > 5) {
    consecutive_failures_ = 0;
    return true;
  }

  return false;
}

std::vector<cv::Point2d> AdaptiveLineTracker::smooth_trajectory(
    const std::vector<cv::Point2d>& points) {
  if (points.size() < config_.smooth_window) {
    return points;
  }

  std::vector<cv::Point2d> smoothed;
  smoothed.reserve(points.size());

  int half_window = config_.smooth_window / 2;

  for (size_t i = 0; i < points.size(); i++) {
    double sum_x = 0;
    double sum_y = 0;
    int count = 0;

    for (int j = -half_window; j <= half_window; j++) {
      int idx = static_cast<int>(i) + j;
      if (idx >= 0 && idx < static_cast<int>(points.size())) {
        sum_x += points[idx].x;
        sum_y += points[idx].y;
        count++;
      }
    }

    if (count > 0) {
      smoothed.push_back(cv::Point2d(sum_x / count, sum_y / count));
    } else {
      smoothed.push_back(points[i]);
    }
  }

  return smoothed;
}

void AdaptiveLineTracker::update_confidence(bool detection_success,
                                            double prediction_error) {
  if (detection_success) {
    // Increase confidence based on prediction accuracy
    double error_factor = std::exp(-prediction_error / 10.0);
    confidence_ = confidence_ * 0.9 + error_factor * 0.1;
  } else {
    // Decrease confidence on failure
    confidence_ *= 0.95;
  }

  // Clamp confidence to [0, 1]
  confidence_ = std::max(0.0, std::min(1.0, confidence_));
}