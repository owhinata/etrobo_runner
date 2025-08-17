#ifndef SIMPLE_LINE_SELECTOR_HPP_
#define SIMPLE_LINE_SELECTOR_HPP_

#include <deque>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

class SimpleLineSelector {
 public:
  struct Segment {
    int start_x;
    int end_x;
    int y;

    cv::Point2f center() const {
      return cv::Point2f((start_x + end_x) / 2.0f, y);
    }
    float width() const { return end_x - start_x; }
  };

  SimpleLineSelector(int history_size = 10);

  // Add new scan line segments
  void add_scan(int y, const std::vector<Segment>& segments);

  // Get the best segment for the current scan
  std::optional<Segment> get_best_segment() const;

  // Reset the selector
  void reset();

 private:
  struct ScanHistory {
    int y;
    std::vector<Segment> segments;
    int selected_index;  // Which segment was selected
  };

  // Calculate continuity score between two segments
  float calculate_continuity(const Segment& prev, const Segment& curr) const;

  // Select best segment based on history
  int select_best_segment(const std::vector<Segment>& segments) const;

 private:
  std::deque<ScanHistory> history_;
  int max_history_size_;

  // Branch handling state
  bool in_branch_mode_ = false;
  int branch_selection_counter_ = 0;
};

#endif  // SIMPLE_LINE_SELECTOR_HPP_