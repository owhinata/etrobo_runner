# etrobo_line_detector

A ROS 2 C++ node that detects straight lines using Canny + Hough transform and publishes both the detection results and a visualization image.

## Features
- Latency-first: never drops the frame being processed; the subscription queue always keeps only the latest frame (QoS depth = 1).
- Flexible parameters: dynamically tune Canny/Hough, pre-processing, and visualization.
- Visualization: publishes an image with detected lines overlaid and RViz `Marker`s.
- Timing logs: outputs per-frame processing time using `steady_clock` at INFO level.

## Dependencies
- rclcpp, sensor_msgs, std_msgs, visualization_msgs
- image_transport, cv_bridge
- OpenCV (core, imgproc, imgcodecs)

## Build
Run the following at the workspace root:

```
colcon build --packages-select etrobo_line_detector
source install/setup.bash
```

## Run
```
ros2 run etrobo_line_detector etrobo_line_detector
```

Common options:
```
ros2 run etrobo_line_detector etrobo_line_detector \
  --ros-args \
  -p image_topic:=/camera/image_raw \
  -p canny_low:=80 -p canny_high:=200 \
  -p hough_type:=probabilistic
```

## Topics
- Input: `~image` (`sensor_msgs/msg/Image`)
- Output:
  - `~image_with_lines` (`sensor_msgs/msg/Image`)
  - `~lines` (`std_msgs/msg/Float32MultiArray`) — lists `[x1, y1, x2, y2]`
  - `~markers` (`visualization_msgs/msg/MarkerArray`)

## Parameters (excerpt)
- I/O: `image_topic`, `use_color_output`
- Pre-processing: `grayscale`, `blur_ksize`, `blur_sigma`, `roi`, `downscale`
- Canny: `canny_low`, `canny_high`, `canny_aperture`, `canny_L2gradient`
- Hough: `hough_type`, `rho`, `theta_deg`, `threshold`, `min_line_length`, `max_line_gap`, `min_theta_deg`, `max_theta_deg`
- Visualization: `draw_color_bgr`, `draw_thickness`, `publish_markers`

See [doc/DESIGN.md](doc/DESIGN.md) for details.

## License
This project follows the repository’s license policy.

