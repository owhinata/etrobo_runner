#ifndef BRANCH_MERGE_HANDLER_HPP_
#define BRANCH_MERGE_HANDLER_HPP_

#include <deque>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

class BranchMergeHandler {
 public:
  // Segment structure to represent a detected line segment
  struct Segment {
    int start_x;
    int end_x;

    double center() const { return (start_x + end_x) / 2.0; }
    double width() const { return end_x - start_x; }
  };

  // Configuration for branch/merge handling strategies
  struct Config {
    enum BranchStrategy {
      LEFT_PRIORITY,      // Select leftmost branch
      RIGHT_PRIORITY,     // Select rightmost branch
      STRAIGHT_PRIORITY,  // Select branch closest to previous direction
      ALTERNATING         // Alternate between right and left for each branch
    };

    enum MergeStrategy {
      CONTINUITY,   // Select based on continuity with previous segment
      WIDTH_BASED,  // Select based on width similarity
      CENTER_BASED  // Select based on center position
    };

    bool enabled = true;
    BranchStrategy branch_strategy = ALTERNATING;
    MergeStrategy merge_strategy = CONTINUITY;
    double continuity_threshold = 30.0;  // Max pixel distance for continuity
  };

  // Context information for processing segments
  struct SegmentContext {
    std::vector<Segment> current_segments;   // Current scan line segments
    std::vector<Segment> previous_segments;  // Previous scan line segments
    int contour_id;
    int scan_y;
    int scan_step;  // Distance between scan lines
  };

  // Line topology classification
  enum class LineTopology {
    SINGLE,  // Normal single line
    MERGE,   // Multiple lines merging into one
    BRANCH,  // Single line branching into multiple
    COMPLEX  // Other complex patterns
  };

  // Cluster for segment analysis
  struct SegmentCluster {
    int start_y;
    int end_y;
    std::vector<std::vector<Segment>> scan_lines;
    int segment_count;  // Number of segments in this cluster (1 or 2+)

    // Statistics
    double avg_segment_distance;
    double distance_trend;  // +: diverging, -: converging
    bool is_stable;
  };

  BranchMergeHandler();
  ~BranchMergeHandler() = default;

  // Main processing function - returns selected segment if any
  std::optional<Segment> process_segments(const SegmentContext& context);

  // Configuration
  void set_config(const Config& config);
  Config get_config() const;

  // Get last detected topology for debugging
  LineTopology get_last_topology() const { return last_topology_; }

  // Reset branch counter (for testing or when starting new sequence)
  void reset_branch_counter() { branch_count_ = 0; }

 private:
  // Detect the topology type from segment counts
  LineTopology detect_topology(const SegmentContext& context) const;

  // Handle branching case - select one path
  std::optional<Segment> handle_branch(const SegmentContext& context) const;

  // Handle merging case - select main line
  std::optional<Segment> handle_merge(const SegmentContext& context) const;

  // Handle single segment case
  std::optional<Segment> handle_single(const SegmentContext& context) const;

  // Select segment based on branch strategy
  Segment select_branch_segment(const std::vector<Segment>& segments,
                                const Segment& previous) const;

  // Select segment based on merge strategy
  Segment select_merge_segment(const std::vector<Segment>& segments,
                               const std::vector<Segment>& previous) const;

  // Calculate continuity score between two segments
  double calculate_continuity_score(const Segment& current,
                                    const Segment& previous,
                                    int y_distance) const;

  // Find best matching previous segment
  std::optional<Segment> find_best_previous_match(
      const Segment& current, const std::vector<Segment>& previous,
      int y_distance) const;

  // Cluster-based analysis
  void update_clusters(int y, const std::vector<Segment>& segments);
  void analyze_cluster_statistics(SegmentCluster& cluster);
  LineTopology detect_topology_from_clusters() const;
  bool should_start_new_cluster(const std::vector<Segment>& segments) const;
  double calculate_segment_distance(const std::vector<Segment>& segments) const;

  Config config_;
  mutable LineTopology last_topology_ = LineTopology::SINGLE;
  mutable int branch_count_ = 0;  // Counter for alternating branch selection

  // Cluster management
  mutable std::deque<SegmentCluster> clusters_;
  static constexpr size_t MAX_CLUSTERS = 5;
  static constexpr size_t MIN_CLUSTER_SIZE = 3;
};

#endif  // BRANCH_MERGE_HANDLER_HPP_