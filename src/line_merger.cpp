#include "src/line_merger.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

LineMerger::LineMerger() {}

std::vector<std::vector<cv::Point>> LineMerger::merge_lines(
    const std::map<int, ContourTracker::TrackedContour>& tracked_contours) {
  if (!config_.enabled || tracked_contours.empty()) {
    // Return original contours without merging
    std::vector<std::vector<cv::Point>> result;
    for (const auto& [id, contour] : tracked_contours) {
      if (contour.missed_frames == 0) {
        result.push_back(contour.contour);
      }
    }
    return result;
  }

  merge_groups_.clear();
  std::vector<MergeCandidate> candidates;

  if (config_.method == MergeMethod::DIRECTION_ENDPOINT) {
    // Method 1+2: Direction and Endpoint based
    std::map<int, LineSegment> line_segments;

    // Convert tracked contours to line segments
    for (const auto& [id, contour] : tracked_contours) {
      if (contour.missed_frames > 0) continue;

      LineSegment segment;
      segment.id = id;
      segment.points = contour.contour;

      if (segment.points.size() < 2) continue;

      // Calculate line properties
      segment.direction = calculate_line_direction(segment.points);
      extract_line_endpoints(segment);
      segment.length = cv::norm(segment.end_point - segment.start_point);

      if (segment.length >= config_.min_line_length) {
        line_segments[id] = segment;
      }
    }

    candidates = find_merge_candidates_direction(line_segments);

  } else {
    // Method 4+5: Kalman and Graph based
    candidates = find_merge_candidates_kalman(tracked_contours);
  }

  // Build merge graph and find connected components
  build_merge_graph(candidates);
  find_connected_components();

  // Create merged contours
  std::vector<std::vector<cv::Point>> merged_contours;
  std::set<int> processed_ids;

  for (const auto& [group_id, member_ids] : merge_groups_) {
    if (member_ids.size() > 1) {
      // Merge multiple contours
      std::vector<std::vector<cv::Point>> contours_to_merge;
      for (int id : member_ids) {
        auto it = tracked_contours.find(id);
        if (it != tracked_contours.end()) {
          contours_to_merge.push_back(it->second.contour);
          processed_ids.insert(id);
        }
      }

      if (!contours_to_merge.empty()) {
        auto merged = merge_point_sets(contours_to_merge);
        if (!merged.empty()) {
          merged_contours.push_back(merged);
        }
      }
    }
  }

  // Add unmerged contours
  for (const auto& [id, contour] : tracked_contours) {
    if (contour.missed_frames == 0 &&
        processed_ids.find(id) == processed_ids.end()) {
      merged_contours.push_back(contour.contour);
    }
  }

  return merged_contours;
}

// Method 1+2: Direction and Endpoint based merging
std::vector<LineMerger::MergeCandidate>
LineMerger::find_merge_candidates_direction(
    const std::map<int, LineSegment>& line_segments) {
  std::vector<MergeCandidate> candidates;

  for (auto it1 = line_segments.begin(); it1 != line_segments.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != line_segments.end(); ++it2) {
      const auto& seg1 = it1->second;
      const auto& seg2 = it2->second;

      // Calculate angle difference
      float angle_diff =
          calculate_angle_difference(seg1.direction, seg2.direction);
      if (angle_diff > config_.max_angle_diff) continue;

      // Calculate endpoint distance
      float endpoint_dist = calculate_endpoint_distance(seg1, seg2);
      if (endpoint_dist > config_.max_endpoint_dist) continue;

      // Calculate merge score (lower angle diff and endpoint dist = higher
      // score)
      float angle_score = 1.0f - (angle_diff / config_.max_angle_diff);
      float dist_score = 1.0f - (endpoint_dist / config_.max_endpoint_dist);
      float merge_score = (angle_score * 0.6f + dist_score * 0.4f);

      MergeCandidate candidate;
      candidate.line1_id = seg1.id;
      candidate.line2_id = seg2.id;
      candidate.score = merge_score;
      candidates.push_back(candidate);
    }
  }

  // Sort by score
  std::sort(candidates.begin(), candidates.end());
  return candidates;
}

cv::Vec2f LineMerger::calculate_line_direction(
    const std::vector<cv::Point>& points) {
  if (points.size() < 2) return cv::Vec2f(0, 0);

  // Convert points to Mat for PCA
  cv::Mat data(points.size(), 2, CV_32F);
  for (size_t i = 0; i < points.size(); ++i) {
    data.at<float>(i, 0) = points[i].x;
    data.at<float>(i, 1) = points[i].y;
  }

  // Perform PCA
  cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

  // Get primary direction
  cv::Vec2f direction(pca.eigenvectors.at<float>(0, 0),
                      pca.eigenvectors.at<float>(0, 1));

  // Normalize
  float norm = cv::norm(direction);
  if (norm > 0) {
    direction /= norm;
  }

  return direction;
}

void LineMerger::extract_line_endpoints(LineSegment& segment) {
  if (segment.points.empty()) return;

  // Find extreme points along the line direction
  float min_proj = std::numeric_limits<float>::max();
  float max_proj = std::numeric_limits<float>::lowest();
  cv::Point2f min_point, max_point;

  for (const auto& pt : segment.points) {
    float projection =
        pt.x * segment.direction[0] + pt.y * segment.direction[1];

    if (projection < min_proj) {
      min_proj = projection;
      min_point = cv::Point2f(pt.x, pt.y);
    }
    if (projection > max_proj) {
      max_proj = projection;
      max_point = cv::Point2f(pt.x, pt.y);
    }
  }

  segment.start_point = min_point;
  segment.end_point = max_point;
}

float LineMerger::calculate_angle_difference(const cv::Vec2f& dir1,
                                             const cv::Vec2f& dir2) {
  float dot = dir1.dot(dir2);
  // Clamp to [-1, 1] to avoid numerical issues
  dot = std::max(-1.0f, std::min(1.0f, dot));

  // Consider both directions (line can go either way)
  float angle1 = std::acos(std::abs(dot)) * 180.0f / CV_PI;
  return angle1;
}

float LineMerger::calculate_endpoint_distance(const LineSegment& seg1,
                                              const LineSegment& seg2) {
  // Calculate all possible endpoint distances
  float dist1 = cv::norm(seg1.end_point - seg2.start_point);
  float dist2 = cv::norm(seg1.start_point - seg2.end_point);
  float dist3 = cv::norm(seg1.end_point - seg2.end_point);
  float dist4 = cv::norm(seg1.start_point - seg2.start_point);

  // Return minimum distance
  return std::min({dist1, dist2, dist3, dist4});
}

// Method 4+5: Kalman and Graph based merging
std::vector<LineMerger::MergeCandidate>
LineMerger::find_merge_candidates_kalman(
    const std::map<int, ContourTracker::TrackedContour>& tracked_contours) {
  std::vector<MergeCandidate> candidates;

  for (auto it1 = tracked_contours.begin(); it1 != tracked_contours.end();
       ++it1) {
    for (auto it2 = std::next(it1); it2 != tracked_contours.end(); ++it2) {
      if (it1->second.missed_frames > 0 || it2->second.missed_frames > 0)
        continue;

      // Check if trajectories will intersect or come close
      bool will_merge = check_trajectory_intersection(
          it1->second, it2->second, config_.prediction_frames);

      if (will_merge) {
        MergeCandidate candidate;
        candidate.line1_id = it1->first;
        candidate.line2_id = it2->first;
        candidate.score = config_.merge_confidence;
        candidates.push_back(candidate);
      }
    }
  }

  return candidates;
}

cv::Point2f LineMerger::predict_future_position(
    const ContourTracker::TrackedContour& contour, int frames_ahead) {
  // Extract velocity from Kalman filter state
  float vx = contour.kf.statePost.at<float>(2);
  float vy = contour.kf.statePost.at<float>(3);

  // Predict future position
  cv::Point2f future_pos;
  future_pos.x = contour.centroid.x + vx * frames_ahead;
  future_pos.y = contour.centroid.y + vy * frames_ahead;

  return future_pos;
}

bool LineMerger::check_trajectory_intersection(
    const ContourTracker::TrackedContour& contour1,
    const ContourTracker::TrackedContour& contour2, int frames_ahead) {
  // Get current and predicted positions
  cv::Point2f curr1 = contour1.centroid;
  cv::Point2f curr2 = contour2.centroid;
  cv::Point2f pred1 = predict_future_position(contour1, frames_ahead);
  cv::Point2f pred2 = predict_future_position(contour2, frames_ahead);

  // Check if trajectories intersect or come close
  // Simplified: check if minimum distance along trajectories is below threshold

  for (int t = 0; t <= frames_ahead; ++t) {
    float alpha = static_cast<float>(t) / frames_ahead;
    cv::Point2f pos1 = curr1 * (1 - alpha) + pred1 * alpha;
    cv::Point2f pos2 = curr2 * (1 - alpha) + pred2 * alpha;

    float dist = cv::norm(pos1 - pos2);
    if (dist < config_.trajectory_threshold) {
      return true;
    }
  }

  return false;
}

void LineMerger::build_merge_graph(
    const std::vector<MergeCandidate>& candidates) {
  // Build adjacency list from candidates
  std::map<int, std::set<int>> adjacency;

  for (const auto& candidate : candidates) {
    if (candidate.score >= config_.merge_confidence) {
      adjacency[candidate.line1_id].insert(candidate.line2_id);
      adjacency[candidate.line2_id].insert(candidate.line1_id);
    }
  }

  // Store as initial merge groups
  merge_groups_.clear();
  for (const auto& [id, neighbors] : adjacency) {
    merge_groups_[id] = neighbors;
    merge_groups_[id].insert(id);  // Include self
  }
}

void LineMerger::find_connected_components() {
  // Use Union-Find to find connected components
  std::map<int, int> parent;

  // Initialize: each node is its own parent
  for (const auto& [id, _] : merge_groups_) {
    parent[id] = id;
  }

  // Find operation with path compression
  std::function<int(int)> find = [&](int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  };

  // Union operation
  auto unite = [&](int x, int y) {
    int px = find(x);
    int py = find(y);
    if (px != py) {
      parent[px] = py;
    }
  };

  // Unite connected nodes
  for (const auto& [id, neighbors] : merge_groups_) {
    for (int neighbor : neighbors) {
      unite(id, neighbor);
    }
  }

  // Group by representative
  std::map<int, std::set<int>> components;
  for (const auto& [id, _] : merge_groups_) {
    int rep = find(id);
    components[rep].insert(id);
  }

  // Update merge groups
  merge_groups_ = components;
}

std::vector<cv::Point> LineMerger::merge_point_sets(
    const std::vector<std::vector<cv::Point>>& point_sets) {
  if (point_sets.empty()) return {};
  if (point_sets.size() == 1) return point_sets[0];

  // Combine all points
  std::vector<cv::Point> all_points;
  for (const auto& points : point_sets) {
    all_points.insert(all_points.end(), points.begin(), points.end());
  }

  // Sort points along the primary direction
  sort_points_along_line(all_points);

  // Optional: Remove duplicates or smooth the merged line
  // (keeping all points for now)

  return all_points;
}

void LineMerger::sort_points_along_line(std::vector<cv::Point>& points) {
  if (points.size() < 2) return;

  // Calculate primary direction
  cv::Vec2f direction = calculate_line_direction(points);

  // Sort points by projection onto primary direction
  std::sort(points.begin(), points.end(),
            [&direction](const cv::Point& a, const cv::Point& b) {
              float proj_a = a.x * direction[0] + a.y * direction[1];
              float proj_b = b.x * direction[0] + b.y * direction[1];
              return proj_a < proj_b;
            });
}