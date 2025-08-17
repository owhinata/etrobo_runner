#include "simple_line_selector.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

SimpleLineSelector::SimpleLineSelector(int history_size)
    : max_history_size_(history_size) {}

void SimpleLineSelector::add_scan(int y, const std::vector<Segment>& segments) {
  ScanHistory scan;
  scan.y = y;
  scan.segments = segments;

  // Select the best segment for this scan
  scan.selected_index = select_best_segment(segments);

  // Add to history
  history_.push_back(scan);

  // Keep history size limited
  while (history_.size() > max_history_size_) {
    history_.pop_front();
  }

  // Detect branch mode
  if (segments.size() >= 2) {
    if (!in_branch_mode_) {
      in_branch_mode_ = true;
      branch_selection_counter_ = 0;
    }
  } else if (segments.size() <= 1 && in_branch_mode_) {
    in_branch_mode_ = false;
  }
}

float SimpleLineSelector::calculate_continuity(const Segment& prev,
                                               const Segment& curr) const {
  // Calculate position continuity
  cv::Point2f prev_center = prev.center();
  cv::Point2f curr_center = curr.center();

  // Expected position (straight line continuation)
  float expected_x = prev_center.x;
  float x_error = std::abs(curr_center.x - expected_x);

  // Width similarity
  float width_ratio = std::min(prev.width(), curr.width()) /
                      std::max(prev.width(), curr.width());

  // Combined score (higher is better)
  float position_score = std::exp(-x_error / 20.0f);  // Exponential decay
  float continuity_score = position_score * width_ratio;

  return continuity_score;
}

int SimpleLineSelector::select_best_segment(
    const std::vector<Segment>& segments) const {
  if (segments.empty()) {
    return -1;
  }

  if (segments.size() == 1) {
    return 0;
  }

  // If we have history, use it to select the best continuation
  if (!history_.empty()) {
    const auto& last_scan = history_.back();
    if (last_scan.selected_index >= 0 &&
        last_scan.selected_index < last_scan.segments.size()) {
      const Segment& last_selected =
          last_scan.segments[last_scan.selected_index];

      // Find best continuation
      int best_idx = 0;
      float best_score = -1;

      for (size_t i = 0; i < segments.size(); i++) {
        float score = calculate_continuity(last_selected, segments[i]);

        // In branch mode, apply alternating selection bias
        if (in_branch_mode_ && segments.size() == 2) {
          // Alternate between left (0) and right (1)
          bool prefer_right = (branch_selection_counter_ % 2 == 0);
          if ((prefer_right && i == 1) || (!prefer_right && i == 0)) {
            score *= 1.2f;  // Slight bias for alternating selection
          }
        }

        if (score > best_score) {
          best_score = score;
          best_idx = i;
        }
      }

      return best_idx;
    }
  }

  // No history - select the segment closest to center or largest
  int best_idx = 0;
  float max_width = segments[0].width();

  for (size_t i = 1; i < segments.size(); i++) {
    if (segments[i].width() > max_width) {
      max_width = segments[i].width();
      best_idx = i;
    }
  }

  return best_idx;
}

std::optional<SimpleLineSelector::Segment>
SimpleLineSelector::get_best_segment() const {
  if (history_.empty()) {
    return std::nullopt;
  }

  const auto& last_scan = history_.back();
  if (last_scan.selected_index >= 0 &&
      last_scan.selected_index < last_scan.segments.size()) {
    return last_scan.segments[last_scan.selected_index];
  }

  return std::nullopt;
}

void SimpleLineSelector::reset() {
  history_.clear();
  in_branch_mode_ = false;
  branch_selection_counter_ = 0;
}