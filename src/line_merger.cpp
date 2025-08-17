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

  if (config_.debug_enabled && !merge_groups_.empty()) {
    printf("\n[LineMerger] === Merge Groups ===\n");
    for (const auto& [group_id, members] : merge_groups_) {
      if (members.size() > 1) {
        printf("[LineMerger] Group %d: {", group_id);
        for (int id : members) {
          printf("%d,", id);
        }
        printf("}\n");
      }
    }
  }

  // Create merged contours and track ID mapping
  std::vector<std::vector<cv::Point>> merged_contours;
  merged_id_mapping_.clear();
  std::set<int> processed_ids;
  int result_index = 0;

  for (const auto& [group_id, member_ids] : merge_groups_) {
    if (member_ids.size() > 1) {
      // Merge multiple contours
      std::vector<std::vector<cv::Point>> contours_to_merge;
      std::set<int> valid_ids;
      for (int id : member_ids) {
        auto it = tracked_contours.find(id);
        if (it != tracked_contours.end()) {
          contours_to_merge.push_back(it->second.contour);
          processed_ids.insert(id);
          valid_ids.insert(id);
        }
      }

      if (!contours_to_merge.empty()) {
        auto merged = merge_point_sets(contours_to_merge);
        if (!merged.empty()) {
          merged_contours.push_back(merged);
          merged_id_mapping_[result_index] = valid_ids;  // Store merged IDs
          result_index++;
        }
      }
    }
  }

  // Add unmerged contours
  for (const auto& [id, contour] : tracked_contours) {
    if (contour.missed_frames == 0 &&
        processed_ids.find(id) == processed_ids.end()) {
      merged_contours.push_back(contour.contour);
      merged_id_mapping_[result_index] = {id};  // Single ID
      result_index++;
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

  if (config_.debug_enabled) {
    printf("\n[LineMerger::Kalman] === Finding merge candidates ===\n");
    printf("[LineMerger::Kalman] Total tracked contours: %zu\n",
           tracked_contours.size());
    printf("[LineMerger::Kalman] Prediction frames: %d\n",
           config_.prediction_frames);
    printf("[LineMerger::Kalman] Trajectory threshold: %.1f\n",
           config_.trajectory_threshold);
  }

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

        if (config_.debug_enabled) {
          printf(
              "[LineMerger::Kalman] Found merge candidate: ID%d <-> ID%d "
              "(score: %.2f)\n",
              candidate.line1_id, candidate.line2_id, candidate.score);
        }
      }
    }
  }

  if (config_.debug_enabled) {
    printf("[LineMerger::Kalman] Total candidates found: %zu\n",
           candidates.size());
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

  if (config_.debug_enabled) {
    float speed = std::sqrt(vx * vx + vy * vy);
    printf(
        "  [Predict] ID%d: Current(%.1f,%.1f) vel(%.2f,%.2f) speed=%.2f -> "
        "Future(%.1f,%.1f)\n",
        contour.id, contour.centroid.x, contour.centroid.y, vx, vy, speed,
        future_pos.x, future_pos.y);
    if (speed < 0.1) {
      printf("  [Predict] WARNING: Very low velocity detected (age=%d)\n",
             contour.age);
    }
  }

  return future_pos;
}

bool LineMerger::check_trajectory_intersection(
    const ContourTracker::TrackedContour& contour1,
    const ContourTracker::TrackedContour& contour2, int frames_ahead) {
  // Get current and predicted positions
  cv::Point2f curr1 = contour1.centroid;
  cv::Point2f curr2 = contour2.centroid;

  if (config_.debug_enabled) {
    printf("\n[LineMerger::Trajectory] Checking ID%d vs ID%d\n", contour1.id,
           contour2.id);
    printf("  [Trajectory] Contour1: age=%d, missed=%d, area=%.1f\n",
           contour1.age, contour1.missed_frames, contour1.area);
    printf("  [Trajectory] Contour2: age=%d, missed=%d, area=%.1f\n",
           contour2.age, contour2.missed_frames, contour2.area);
  }

  // First check line continuity based on geometric properties
  bool continuous = check_line_continuity(contour1, contour2);
  if (continuous) {
    if (config_.debug_enabled) {
      printf("  [Trajectory] WILL MERGE (line continuity check passed)\n");
    }
    return true;
  }

  cv::Point2f pred1 = predict_future_position(contour1, frames_ahead);
  cv::Point2f pred2 = predict_future_position(contour2, frames_ahead);

  // Check if trajectories intersect or come close
  // Simplified: check if minimum distance along trajectories is below threshold

  float min_dist = std::numeric_limits<float>::max();
  int min_frame = -1;
  float initial_dist = cv::norm(curr1 - curr2);

  // Extract velocities for analysis
  float vx1 = contour1.kf.statePost.at<float>(2);
  float vy1 = contour1.kf.statePost.at<float>(3);
  float vx2 = contour2.kf.statePost.at<float>(2);
  float vy2 = contour2.kf.statePost.at<float>(3);
  float speed1 = std::sqrt(vx1 * vx1 + vy1 * vy1);
  float speed2 = std::sqrt(vx2 * vx2 + vy2 * vy2);

  // If both have very low velocity, use static distance check
  if (speed1 < 0.5 && speed2 < 0.5 && contour1.age > 1 && contour2.age > 1) {
    // For static contours, check if they're close enough to be the same line
    if (initial_dist <
        config_.trajectory_threshold * config_.static_distance_multiplier) {
      if (config_.debug_enabled) {
        printf(
            "  [Trajectory] WILL MERGE (static proximity): dist=%.1f < "
            "threshold=%.1f\n",
            initial_dist,
            config_.trajectory_threshold * config_.static_distance_multiplier);
      }
      return true;
    }
  }

  for (int t = 0; t <= frames_ahead; ++t) {
    float alpha = static_cast<float>(t) / frames_ahead;
    cv::Point2f pos1 = curr1 * (1 - alpha) + pred1 * alpha;
    cv::Point2f pos2 = curr2 * (1 - alpha) + pred2 * alpha;

    float dist = cv::norm(pos1 - pos2);
    if (dist < min_dist) {
      min_dist = dist;
      min_frame = t;
    }

    if (dist < config_.trajectory_threshold) {
      if (config_.debug_enabled) {
        printf(
            "  [Trajectory] WILL MERGE at frame %d: dist=%.1f < "
            "threshold=%.1f\n",
            t, dist, config_.trajectory_threshold);
        printf("  [Trajectory] Initial dist=%.1f, Min dist=%.1f at frame %d\n",
               initial_dist, min_dist, min_frame);
      }
      return true;
    }
  }

  if (config_.debug_enabled) {
    printf(
        "  [Trajectory] NO MERGE: min_dist=%.1f at frame %d > threshold=%.1f\n",
        min_dist, min_frame, config_.trajectory_threshold);
    printf("  [Trajectory] Initial dist=%.1f, velocity divergence\n",
           initial_dist);
  }

  return false;
}

void LineMerger::build_merge_graph(
    const std::vector<MergeCandidate>& candidates) {
  // Build adjacency list from candidates
  std::map<int, std::set<int>> adjacency;

  if (config_.debug_enabled) {
    printf("\n[LineMerger::Graph] Building merge graph from %zu candidates\n",
           candidates.size());
    printf("[LineMerger::Graph] Merge confidence threshold: %.2f\n",
           config_.merge_confidence);
  }

  int accepted_count = 0;
  for (const auto& candidate : candidates) {
    if (candidate.score >= config_.merge_confidence) {
      adjacency[candidate.line1_id].insert(candidate.line2_id);
      adjacency[candidate.line2_id].insert(candidate.line1_id);
      accepted_count++;

      if (config_.debug_enabled) {
        printf(
            "[LineMerger::Graph] Accept edge: ID%d <-> ID%d (score %.2f >= "
            "%.2f)\n",
            candidate.line1_id, candidate.line2_id, candidate.score,
            config_.merge_confidence);
      }
    } else if (config_.debug_enabled) {
      printf(
          "[LineMerger::Graph] Reject edge: ID%d <-> ID%d (score %.2f < "
          "%.2f)\n",
          candidate.line1_id, candidate.line2_id, candidate.score,
          config_.merge_confidence);
    }
  }

  if (config_.debug_enabled) {
    printf("[LineMerger::Graph] Accepted %d/%zu edges\n", accepted_count,
           candidates.size());
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

// Line continuity checking functions
bool LineMerger::check_line_continuity(
    const ContourTracker::TrackedContour& contour1,
    const ContourTracker::TrackedContour& contour2) {
  // Skip if contours are too small
  if (contour1.contour.size() < 5 || contour2.contour.size() < 5) {
    return false;
  }

  // Fit lines to both contours
  cv::Vec4f line1 = fit_line_to_contour(contour1.contour);
  cv::Vec4f line2 = fit_line_to_contour(contour2.contour);

  // Check if lines are collinear (similar direction and close distance)
  float angle_threshold = 15.0f;     // degrees
  float distance_threshold = 20.0f;  // pixels

  bool collinear =
      are_lines_collinear(line1, line2, angle_threshold, distance_threshold);

  if (!collinear) {
    if (config_.debug_enabled) {
      printf("  [Continuity] Lines are not collinear\n");
    }
    return false;
  }

  // Check the gap between the lines
  float gap =
      get_gap_between_lines(contour1.contour, contour2.contour, line1, line2);

  if (config_.debug_enabled) {
    printf("  [Continuity] Lines are collinear, gap=%.1f\n", gap);
  }

  // Accept if gap is reasonable
  // For vertical lines, we want a small positive gap or small overlap
  // Large gaps mean disconnected lines, large overlaps mean different parallel
  // lines
  if (gap < 0) {
    // Overlap case - accept only small overlaps
    return gap > -50.0;  // Allow up to 50 pixel overlap
  } else {
    // Gap case - accept if within threshold
    return gap < config_.trajectory_threshold;
  }
}

cv::Vec4f LineMerger::fit_line_to_contour(
    const std::vector<cv::Point>& contour) {
  cv::Vec4f line;
  cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
  return line;
}

float LineMerger::point_to_line_distance(const cv::Point2f& point,
                                         const cv::Vec4f& line) {
  // Line is represented as (vx, vy, x0, y0) where (vx, vy) is direction
  // and (x0, y0) is a point on the line
  cv::Point2f line_point(line[2], line[3]);
  cv::Vec2f line_dir(line[0], line[1]);

  // Vector from line point to the point
  cv::Vec2f to_point(point.x - line_point.x, point.y - line_point.y);

  // Project onto line direction
  float proj = to_point.dot(line_dir);

  // Get perpendicular distance
  cv::Point2f on_line =
      line_point + proj * cv::Point2f(line_dir[0], line_dir[1]);
  return cv::norm(point - on_line);
}

bool LineMerger::are_lines_collinear(const cv::Vec4f& line1,
                                     const cv::Vec4f& line2,
                                     float angle_threshold,
                                     float distance_threshold) {
  // Check angle between lines
  cv::Vec2f dir1(line1[0], line1[1]);
  cv::Vec2f dir2(line2[0], line2[1]);

  float dot =
      std::abs(dir1.dot(dir2));  // Use abs to handle opposite directions
  float angle = std::acos(std::min(1.0f, dot)) * 180.0f / CV_PI;

  if (angle > angle_threshold) {
    if (config_.debug_enabled) {
      printf("  [Collinear] Angle difference %.1f > %.1f\n", angle,
             angle_threshold);
    }
    return false;
  }

  // Check distance between lines
  cv::Point2f point1(line1[2], line1[3]);
  cv::Point2f point2(line2[2], line2[3]);

  float dist1 = point_to_line_distance(point1, line2);
  float dist2 = point_to_line_distance(point2, line1);
  float avg_dist = (dist1 + dist2) / 2.0f;

  if (config_.debug_enabled) {
    printf("  [Collinear] Angle=%.1f, Avg distance=%.1f\n", angle, avg_dist);
  }

  return avg_dist < distance_threshold;
}

cv::Point2f LineMerger::project_point_on_line(const cv::Point2f& point,
                                              const cv::Vec4f& line) {
  cv::Point2f line_point(line[2], line[3]);
  cv::Vec2f line_dir(line[0], line[1]);

  cv::Vec2f to_point(point.x - line_point.x, point.y - line_point.y);
  float proj = to_point.dot(line_dir);

  cv::Point2f offset(proj * line_dir[0], proj * line_dir[1]);
  return cv::Point2f(line_point.x + offset.x, line_point.y + offset.y);
}

float LineMerger::get_gap_between_lines(const std::vector<cv::Point>& contour1,
                                        const std::vector<cv::Point>& contour2,
                                        const cv::Vec4f& line1,
                                        const cv::Vec4f& line2) {
  // Find extreme points along the line direction
  cv::Vec2f dir1(line1[0], line1[1]);
  cv::Vec2f dir2(line2[0], line2[1]);

  // Use the direction with larger vertical component for vertical lines
  cv::Vec2f dir;
  if (std::abs(dir1[1]) > std::abs(dir1[0])) {
    // More vertical than horizontal - use Y-axis projection
    dir = cv::Vec2f(0, 1);
  } else {
    // More horizontal - use average direction
    dir = cv::Vec2f((dir1[0] + dir2[0]) / 2, (dir1[1] + dir2[1]) / 2);
    dir /= cv::norm(dir);  // Normalize
  }

  float min1 = std::numeric_limits<float>::max();
  float max1 = std::numeric_limits<float>::lowest();
  float min2 = std::numeric_limits<float>::max();
  float max2 = std::numeric_limits<float>::lowest();

  // Project all points onto the common direction
  for (const auto& pt : contour1) {
    float proj = pt.x * dir[0] + pt.y * dir[1];
    min1 = std::min(min1, proj);
    max1 = std::max(max1, proj);
  }

  for (const auto& pt : contour2) {
    float proj = pt.x * dir[0] + pt.y * dir[1];
    min2 = std::min(min2, proj);
    max2 = std::max(max2, proj);
  }

  // Calculate gap (positive means there's a gap, negative means overlap)
  float gap = std::max(min1 - max2, min2 - max1);

  // For vertical lines, also check actual Y coordinates
  if (std::abs(dir[1]) > 0.9) {  // Nearly vertical
    cv::Rect bbox1 = cv::boundingRect(contour1);
    cv::Rect bbox2 = cv::boundingRect(contour2);

    // Calculate the minimum gap between bounding boxes
    // Positive = gap, Negative = overlap
    float gap_top_to_bottom =
        bbox1.y - (bbox2.y + bbox2.height);  // Top of 1 to bottom of 2
    float gap_bottom_to_top =
        bbox2.y - (bbox1.y + bbox1.height);  // Top of 2 to bottom of 1
    float y_gap = std::max(gap_top_to_bottom, gap_bottom_to_top);

    if (config_.debug_enabled) {
      printf("  [Gap] Projection gap: %.1f, Y-axis gap: %.1f (using Y-axis)\n",
             gap, y_gap);
      printf("  [Gap] BBox1: y=[%d,%d], BBox2: y=[%d,%d]\n", bbox1.y,
             bbox1.y + bbox1.height, bbox2.y, bbox2.y + bbox2.height);
    }

    return y_gap;  // Use actual Y gap for vertical lines
  }

  if (config_.debug_enabled) {
    printf(
        "  [Gap] Contour1: [%.1f, %.1f], Contour2: [%.1f, %.1f], Gap: %.1f\n",
        min1, max1, min2, max2, gap);
  }

  return gap;
}