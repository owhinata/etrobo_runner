#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "src/line_detector_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LineDetectorNode>());
  rclcpp::shutdown();
  return 0;
}