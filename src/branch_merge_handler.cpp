#include "branch_merge_handler.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

BranchMergeHandler::BranchMergeHandler() : config_() {}

std::optional<BranchMergeHandler::Segment> BranchMergeHandler::process_segments(
    const SegmentContext& context) {
  if (!config_.enabled) {
    // If disabled, return the first segment if available
    if (!context.current_segments.empty()) {
      return context.current_segments[0];
    }
    return std::nullopt;
  }

  // Update cluster analysis
  update_clusters(context.scan_y, context.current_segments);

  // Detect topology using cluster-based analysis
  LineTopology topology = detect_topology_from_clusters();

  // Fallback to simple detection if clusters are not ready
  if (topology == LineTopology::SINGLE && clusters_.size() < 2) {
    topology = detect_topology(context);
  }

  last_topology_ = topology;

  // Log topology detection result
  std::string topology_str;
  switch (topology) {
    case LineTopology::BRANCH:
      topology_str = "BRANCH";
      break;
    case LineTopology::MERGE:
      topology_str = "MERGE";
      break;
    case LineTopology::SINGLE:
      topology_str = "SINGLE";
      break;
    case LineTopology::COMPLEX:
      topology_str = "COMPLEX";
      break;
  }

  // Handle based on topology
  std::optional<Segment> result;
  switch (topology) {
    case LineTopology::BRANCH:
      result = handle_branch(context);
      break;
    case LineTopology::MERGE:
      result = handle_merge(context);
      break;
    case LineTopology::SINGLE:
      result = handle_single(context);
      break;
    case LineTopology::COMPLEX:
      // For complex cases, try to use merge logic
      result = handle_merge(context);
      break;
  }

  return result;
}

BranchMergeHandler::LineTopology BranchMergeHandler::detect_topology(
    const SegmentContext& context) const {
  size_t current_count = context.current_segments.size();
  size_t previous_count = context.previous_segments.size();

  if (current_count == 0) {
    return LineTopology::SINGLE;
  }

  if (current_count == 1 && previous_count <= 1) {
    return LineTopology::SINGLE;
  }

  if (previous_count == 1 && current_count > 1) {
    return LineTopology::BRANCH;
  }

  if (previous_count > 1 && current_count == 1) {
    return LineTopology::MERGE;
  }

  return LineTopology::COMPLEX;
}

std::optional<BranchMergeHandler::Segment> BranchMergeHandler::handle_branch(
    const SegmentContext& context) const {
  if (context.current_segments.empty()) {
    return std::nullopt;
  }

  // For single segment, just return it (branch already completed)
  if (context.current_segments.size() == 1) {
    return context.current_segments[0];
  }

  if (context.previous_segments.empty()) {
    // No previous reference, use first segment
    return context.current_segments[0];
  }

  // Select branch based on strategy - only when we have multiple segments
  Segment selected = select_branch_segment(context.current_segments,
                                           context.previous_segments[0]);
  return selected;
}

std::optional<BranchMergeHandler::Segment> BranchMergeHandler::handle_merge(
    const SegmentContext& context) const {
  if (context.current_segments.empty()) {
    return std::nullopt;
  }

  if (context.current_segments.size() == 1) {
    // Already merged to single segment
    return context.current_segments[0];
  }

  // Multiple segments - select main line
  Segment selected =
      select_merge_segment(context.current_segments, context.previous_segments);
  return selected;
}

std::optional<BranchMergeHandler::Segment> BranchMergeHandler::handle_single(
    const SegmentContext& context) const {
  if (context.current_segments.empty()) {
    return std::nullopt;
  }
  return context.current_segments[0];
}

BranchMergeHandler::Segment BranchMergeHandler::select_branch_segment(
    const std::vector<Segment>& segments, const Segment& previous) const {
  if (segments.empty()) {
    return Segment{0, 0};
  }

  if (segments.size() == 1) {
    return segments[0];
  }

  switch (config_.branch_strategy) {
    case Config::LEFT_PRIORITY:
      // Select leftmost segment
      return *std::min_element(segments.begin(), segments.end(),
                               [](const Segment& a, const Segment& b) {
                                 return a.center() < b.center();
                               });

    case Config::RIGHT_PRIORITY:
      // Select rightmost segment
      return *std::max_element(segments.begin(), segments.end(),
                               [](const Segment& a, const Segment& b) {
                                 return a.center() < b.center();
                               });

    case Config::ALTERNATING: {
      // Alternate between right and left for each branch detection
      // Even count (0, 2, 4...): select right
      // Odd count (1, 3, 5...): select left
      bool select_right = (branch_count_ % 2 == 0);
      branch_count_++;  // Increment for next branch

      if (select_right) {
        // Select rightmost segment
        auto result = *std::max_element(segments.begin(), segments.end(),
                                        [](const Segment& a, const Segment& b) {
                                          return a.center() < b.center();
                                        });
        return result;
      } else {
        // Select leftmost segment
        auto result = *std::min_element(segments.begin(), segments.end(),
                                        [](const Segment& a, const Segment& b) {
                                          return a.center() < b.center();
                                        });
        return result;
      }
    }

    case Config::STRAIGHT_PRIORITY:
    default:
      // Select segment closest to previous center
      double prev_center = previous.center();
      return *std::min_element(
          segments.begin(), segments.end(),
          [prev_center](const Segment& a, const Segment& b) {
            return std::abs(a.center() - prev_center) <
                   std::abs(b.center() - prev_center);
          });
  }
}

BranchMergeHandler::Segment BranchMergeHandler::select_merge_segment(
    const std::vector<Segment>& segments,
    const std::vector<Segment>& previous) const {
  if (segments.empty()) {
    return Segment{0, 0};
  }

  if (segments.size() == 1) {
    return segments[0];
  }

  // Default to first segment if no previous segments
  if (previous.empty()) {
    return segments[0];
  }

  switch (config_.merge_strategy) {
    case Config::WIDTH_BASED: {
      // Select segment with width closest to previous segments average
      double avg_prev_width = 0;
      for (const auto& prev : previous) {
        avg_prev_width += prev.width();
      }
      avg_prev_width /= previous.size();

      return *std::min_element(
          segments.begin(), segments.end(),
          [avg_prev_width](const Segment& a, const Segment& b) {
            return std::abs(a.width() - avg_prev_width) <
                   std::abs(b.width() - avg_prev_width);
          });
    }

    case Config::CENTER_BASED: {
      // Select segment closest to center of previous segments
      double avg_prev_center = 0;
      for (const auto& prev : previous) {
        avg_prev_center += prev.center();
      }
      avg_prev_center /= previous.size();

      return *std::min_element(
          segments.begin(), segments.end(),
          [avg_prev_center](const Segment& a, const Segment& b) {
            return std::abs(a.center() - avg_prev_center) <
                   std::abs(b.center() - avg_prev_center);
          });
    }

    case Config::CONTINUITY:
    default: {
      // Select segment with best continuity score
      double best_score = -std::numeric_limits<double>::infinity();
      Segment best_segment = segments[0];

      for (const auto& seg : segments) {
        double score = 0;
        for (const auto& prev : previous) {
          score += calculate_continuity_score(seg, prev, 1);
        }
        if (score > best_score) {
          best_score = score;
          best_segment = seg;
        }
      }
      return best_segment;
    }
  }
}

double BranchMergeHandler::calculate_continuity_score(
    const Segment& current, const Segment& previous, int /*y_distance*/) const {
  // Calculate continuity based on:
  // 1. Center distance
  // 2. Width similarity
  // 3. Overlap

  double center_dist = std::abs(current.center() - previous.center());
  double width_diff = std::abs(current.width() - previous.width());

  // Check overlap
  double overlap_start = std::max(current.start_x, previous.start_x);
  double overlap_end = std::min(current.end_x, previous.end_x);
  double overlap = std::max(0.0, overlap_end - overlap_start);

  // Calculate score (higher is better)
  double score = 0;

  // Penalize center distance
  score -= center_dist / config_.continuity_threshold;

  // Penalize width difference
  score -= width_diff / previous.width() * 0.5;

  // Reward overlap
  score += overlap / previous.width() * 2.0;

  return score;
}

std::optional<BranchMergeHandler::Segment>
BranchMergeHandler::find_best_previous_match(
    const Segment& current, const std::vector<Segment>& previous,
    int y_distance) const {
  if (previous.empty()) {
    return std::nullopt;
  }

  double best_score = -std::numeric_limits<double>::infinity();
  Segment best_match = previous[0];

  for (const auto& prev : previous) {
    double score = calculate_continuity_score(current, prev, y_distance);
    if (score > best_score) {
      best_score = score;
      best_match = prev;
    }
  }

  // Only return if score meets threshold
  if (best_score > -1.0) {
    return best_match;
  }

  return std::nullopt;
}

void BranchMergeHandler::set_config(const Config& config) { config_ = config; }

BranchMergeHandler::Config BranchMergeHandler::get_config() const {
  return config_;
}

// Cluster-based analysis implementation
void BranchMergeHandler::update_clusters(int y,
                                         const std::vector<Segment>& segments) {
  // Check if we should start a new cluster
  if (clusters_.empty() || should_start_new_cluster(segments)) {
    // Start new cluster
    SegmentCluster new_cluster;
    new_cluster.start_y = y;
    new_cluster.end_y = y;
    new_cluster.scan_lines.push_back(segments);
    new_cluster.segment_count = segments.size();
    new_cluster.avg_segment_distance = 0;
    new_cluster.distance_trend = 0;
    new_cluster.is_stable = false;

    clusters_.push_back(new_cluster);

    // Keep only recent clusters
    while (clusters_.size() > MAX_CLUSTERS) {
      clusters_.pop_front();
    }
  } else {
    // Add to existing cluster
    clusters_.back().scan_lines.push_back(segments);
    clusters_.back().end_y = y;

    // Update statistics for the current cluster
    analyze_cluster_statistics(clusters_.back());
  }
}

bool BranchMergeHandler::should_start_new_cluster(
    const std::vector<Segment>& segments) const {
  if (clusters_.empty()) {
    return true;
  }

  const auto& last_cluster = clusters_.back();
  if (last_cluster.scan_lines.empty()) {
    return true;
  }

  // Check if segment count changed
  int last_segment_count = last_cluster.scan_lines.back().size();
  int current_segment_count = segments.size();

  return last_segment_count != current_segment_count;
}

void BranchMergeHandler::analyze_cluster_statistics(SegmentCluster& cluster) {
  if (cluster.scan_lines.size() < 2) {
    return;
  }

  std::vector<double> distances;
  std::vector<double> distance_changes;

  // Calculate distances between segments for each scan line
  for (const auto& scan_line : cluster.scan_lines) {
    if (scan_line.size() >= 2) {
      double max_dist = calculate_segment_distance(scan_line);
      distances.push_back(max_dist);
    }
  }

  // Calculate distance trend
  if (distances.size() >= 2) {
    for (size_t i = 1; i < distances.size(); i++) {
      distance_changes.push_back(distances[i] - distances[i - 1]);
    }

    // Average distance
    cluster.avg_segment_distance =
        std::accumulate(distances.begin(), distances.end(), 0.0) /
        distances.size();

    // Distance trend (positive = diverging, negative = converging)
    if (!distance_changes.empty()) {
      cluster.distance_trend = std::accumulate(distance_changes.begin(),
                                               distance_changes.end(), 0.0) /
                               distance_changes.size();
    }

    // Check stability (small variance in distances)
    double variance = 0;
    for (double d : distances) {
      variance += (d - cluster.avg_segment_distance) *
                  (d - cluster.avg_segment_distance);
    }
    variance /= distances.size();

    cluster.is_stable =
        (variance < 25.0) && (cluster.scan_lines.size() >= MIN_CLUSTER_SIZE);
  }
}

double BranchMergeHandler::calculate_segment_distance(
    const std::vector<Segment>& segments) const {
  if (segments.size() < 2) {
    return 0;
  }

  // Calculate maximum distance between consecutive segments
  double max_distance = 0;
  for (size_t i = 0; i < segments.size() - 1; i++) {
    double gap = segments[i + 1].start_x - segments[i].end_x;
    max_distance = std::max(max_distance, gap);
  }

  return max_distance;
}

BranchMergeHandler::LineTopology
BranchMergeHandler::detect_topology_from_clusters() const {
  if (clusters_.size() < 2) {
    return LineTopology::SINGLE;
  }

  // Analyze the last two clusters - ensure we're looking at the correct
  // clusters
  size_t prev_idx = clusters_.size() - 2;
  size_t curr_idx = clusters_.size() - 1;

  // Find the actual previous cluster with different segment count
  const auto& curr_cluster = clusters_[curr_idx];
  const SegmentCluster* prev_cluster_ptr = nullptr;

  for (int i = clusters_.size() - 2; i >= 0; i--) {
    if (clusters_[i].segment_count != curr_cluster.segment_count) {
      prev_cluster_ptr = &clusters_[i];
      break;
    }
  }

  // If no different cluster found, use the immediate previous one
  if (!prev_cluster_ptr) {
    prev_cluster_ptr = &clusters_[prev_idx];
  }

  const auto& prev_cluster = *prev_cluster_ptr;

  // Need stable clusters for reliable detection
  if (!prev_cluster.is_stable || !curr_cluster.is_stable) {
    // If current cluster is too small, wait for more data
    if (curr_cluster.scan_lines.size() < MIN_CLUSTER_SIZE) {
      return LineTopology::SINGLE;
    }
  }

  // Transition from 1 segment to 2+ segments
  if (prev_cluster.segment_count == 1 && curr_cluster.segment_count >= 2) {
    // Check the trend in the multi-segment cluster
    if (curr_cluster.distance_trend > 2.0) {
      // Segments are diverging -> this is a branch point
      return LineTopology::BRANCH;
    } else if (curr_cluster.distance_trend < -2.0) {
      // Segments are converging -> this is a merge (lines coming together)
      return LineTopology::MERGE;
    }
    // If trend is neutral, need more data
    return LineTopology::SINGLE;
  }

  // Transition from 2+ segments to 1 segment
  if (prev_cluster.segment_count >= 2 && curr_cluster.segment_count == 1) {
    // Check the trend in the previous multi-segment cluster
    if (prev_cluster.distance_trend < -2.0) {
      // Segments were converging -> merge completed
      return LineTopology::MERGE;
    } else if (prev_cluster.distance_trend > 2.0) {
      // Segments were diverging -> branch completed
      return LineTopology::BRANCH;
    }
  }

  // Multi-segment cluster analysis
  if (curr_cluster.segment_count >= 2 && curr_cluster.is_stable) {
    if (curr_cluster.distance_trend < -3.0) {
      // Strong convergence -> merging
      return LineTopology::MERGE;
    } else if (curr_cluster.distance_trend > 3.0) {
      // Strong divergence -> branching
      return LineTopology::BRANCH;
    }
  }

  return LineTopology::SINGLE;
}