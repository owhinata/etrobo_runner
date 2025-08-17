#ifndef ETROBO_LINE_DETECTOR__LINE_DETECTOR_NODE_HPP_
#define ETROBO_LINE_DETECTOR__LINE_DETECTOR_NODE_HPP_

#include <memory>
#include <rclcpp/rclcpp.hpp>

class LineDetectorNode : public rclcpp::Node {
 public:
  enum class State { Calibrating, Localizing };

  // Camera intrinsics structure
  struct CameraIntrinsics {
    bool valid = false;
    double fx = 1.0;
    double fy = 1.0;
    double cx = 0.0;
    double cy = 0.0;
  };

  LineDetectorNode();
  ~LineDetectorNode();

  // Get camera intrinsics for CameraCalibrator
  CameraIntrinsics get_camera_intrinsics() const;

 private:
  // Pimpl idiom
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

#endif  // ETROBO_LINE_DETECTOR__LINE_DETECTOR_NODE_HPP_