# etrobo_line_detector

A ROS 2 C++ node that detects straight lines using Canny + Hough transform and publishes both the detection results and a visualization image. Includes a Python GUI tool for real-time parameter tuning.

## Features
- **Line Detection**: Canny edge detection + Hough transform
- **Startup Calibration**: Estimates camera pitch using a known landmark distance, then switches to normal publishing
- **Low Latency**: Never drops the frame being processed; subscription queue keeps only the latest frame (QoS depth = 1)
- **Flexible Parameters**: Dynamically tune Canny/Hough, pre-processing, and visualization parameters
- **Multiple Outputs**: Publishes detection results as arrays and visualization images
- **Parameter GUI**: Interactive Python GUI for real-time parameter adjustment with live image feed
- **Timing Logs**: Outputs per-frame processing time using `steady_clock` at INFO level

## Dependencies

### C++ Node
- rclcpp, sensor_msgs, std_msgs, visualization_msgs
- image_transport, cv_bridge
- OpenCV (core, imgproc, imgcodecs)

### Python GUI (Optional)
- Python 3.10+
- tkinter (usually included with Python)
- Pillow >= 9.0.1 (for `Image.LANCZOS` compatibility)
- OpenCV Python (`cv2`)
- rclpy, cv_bridge

## Build
Run the following at the workspace root:

```
colcon build --packages-select etrobo_line_detector
source install/setup.bash
```

## Run

### Basic Usage
```bash
ros2 run etrobo_line_detector etrobo_line_detector
```

### With Parameters
```bash
ros2 run etrobo_line_detector etrobo_line_detector \
  --ros-args \
  -p image_topic:=/camera/image_raw \
  -p camera_info_topic:=/camera/camera_info \
  -p canny_low:=80 -p canny_high:=200 \
  -p hough_type:=probabilistic \
  -p publish_image_with_lines:=true
```

### Parameter Tuning GUI
Launch the interactive parameter tuning GUI (requires image output enabled):

```bash
# Terminal 1: Start line detector with image output
ros2 run etrobo_line_detector etrobo_line_detector --ros-args -p publish_image_with_lines:=true

# Terminal 2: Start GUI
cd /path/to/etrobo_line_detector
python3 scripts/line_detector_gui.py
```

### Calibration Tuning
For gray disk detection issues during calibration, use the GUI's **Calibration** section:

```bash
# Start node with continuous calibration (no timeout)
ros2 run etrobo_line_detector etrobo_line_detector \
  --ros-args \
  -p publish_image_with_lines:=true \
  -p calib_timeout_sec:=0.0

# Launch GUI and adjust:
# - calib_hsv_s_max (default=16): Very low saturation for gray objects
# - calib_hsv_v_min (default=100): Minimum brightness for detection
# - calib_hsv_v_max (default=168): Maximum brightness threshold
# - calib_roi_* (default=200,150,240,180): Focus detection region
# - HSV mask visualization appears in top-right corner during calibration
python3 scripts/line_detector_gui.py
```

The GUI provides:
- **Live Image Display**: Real-time visualization of detection results
- **Organized Parameters**: 9 categories of parameters with intuitive controls
- **Interactive Widgets**: Sliders, checkboxes, dropdowns for different parameter types
- **Real-time Updates**: Changes apply immediately to the running node
- **Anti-flicker**: Smooth 30fps display with optimized rendering
- **Calibration Support**: Interactive tuning of gray disk detection parameters during calibration with HSV mask visualization

### Contour Tracking Tips

The line detector includes temporal contour tracking with Kalman filtering. For optimal performance on different track types:

```bash
# For curves and complex tracks (more adaptive tracking)
ros2 run etrobo_line_detector etrobo_line_detector --ros-args \
  -p tracker_process_noise:=0.02 \
  -p tracker_measurement_noise:=0.03 \
  -p tracker_speed_threshold:=3.0 \
  -p tracker_max_distance:=100.0

# For straight tracks (more stable tracking)
ros2 run etrobo_line_detector etrobo_line_detector --ros-args \
  -p tracker_process_noise:=0.005 \
  -p tracker_measurement_noise:=0.08 \
  -p tracker_speed_threshold:=7.0 \
  -p tracker_max_distance:=50.0
```

Key tracking parameters:
- **tracker_process_noise**: Higher values make tracking more responsive but noisier (default: 0.01)
- **tracker_measurement_noise**: Lower values trust measurements more (default: 0.05)
- **tracker_speed_threshold**: Speed (pixels/frame) above which matching distance increases (default: 5.0)
- **tracker_max_distance**: Maximum pixel distance for contour matching (default: 75.0)

## Topics
- **Input**: `~image` (`sensor_msgs/msg/Image`) — source camera image
  - `~camera_info` (`sensor_msgs/msg/CameraInfo`) — camera intrinsics for calibration
- **Output**:
  - `/image_with_lines` (`sensor_msgs/msg/Image`) — visualization with detected lines overlaid
  - `/lines` (`std_msgs/msg/Float32MultiArray`) — line segments as flat `[x1, y1, x2, y2, ...]` array  


## Parameters (excerpt)
- **I/O**: `image_topic`, `publish_image_with_lines`, `show_edges`
- **Calibration**: `camera_info_topic` (string), `camera_height_meters` (double; default 0.2), `landmark_distance_meters` (double; default 0.59), `calib_timeout_sec` (double; default 60.0; set 0 to disable timeout and keep calibrating continuously), `calib_roi` (int[4]; default [200,150,240,180]), HSV thresholds (`calib_hsv_s_max`=16, `calib_hsv_v_min`=100, `calib_hsv_v_max`=168)
- **Pre-processing**: `blur_ksize`, `blur_sigma`, `roi`
- **Canny Edge**: `canny_low`, `canny_high`, `canny_aperture`, `canny_L2gradient`
- **Hough Transform**: `hough_type`, `rho`, `theta_deg`, `threshold`, `min_line_length`, `max_line_gap`
- **HSV Mask**: `use_hsv_mask`, `hsv_lower_*`, `hsv_upper_*`, morphology parameters
- **Visualization**: `draw_color_bgr`, `draw_thickness`

See [doc/DESIGN.md](doc/DESIGN.md) for complete parameter documentation.

## Startup Calibration
- On startup the node enters CalibratePitch state:
  - Does not publish the `lines` topic.
  - Detects the gray start-circle and collects several detections.
  - Computes camera pitch using camera intrinsics, known camera height, and known landmark distance.
  - Transitions to Ready and starts publishing `lines`.
  - If calibration times out (`calib_timeout_sec`), the node proceeds with a 0 rad pitch.

## GUI Parameter Categories
The Python GUI organizes parameters into 7 intuitive sections:
1. **I/O Settings** — Input/output configuration
2. **Pre-processing** — Image preparation (ROI, blur, scaling)
3. **Canny Edge** — Edge detection parameters
4. **Hough Transform** — Line detection settings
5. **HSV Mask** — Color-based filtering for black lines
6. **Calibration** — Camera pitch estimation and gray disk detection settings
7. **Visualization** — Drawing and output options

## License
This project follows the repository’s license policy.
