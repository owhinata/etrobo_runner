# etrobo_line_detector Design Overview

## Overview
- Node name: `etrobo_line_detector`
- Purpose: subscribe to images, detect straight lines using Canny + Hough transform, and publish the results (as an array) and a visualization image with the lines overlaid.
- Execution mode: subscribes to topics only (no file mode).
- Latency policy: prioritize the latest image. Do not abort a frame once processing has started; always finish the current frame.

## Startup Calibration
- States:
  - `CalibratePitch` (initial): enable temporal smoothing; do not publish `~lines`; estimate camera pitch from a gray circle at a known ground distance using `~camera_info` intrinsics.
  - `Ready`: disable temporal smoothing; publish `~lines` and `~markers` normally.
- Parameters:
  - `camera_info_topic` (string; default: `camera_info`)
  - `camera_height_meters` (double; default: `0.2`)
  - `landmark_distance_meters` (double; default: `0.7`)
  - `calib_timeout_sec` (double; default: `60.0`)
  - Uses an internal buffer of landmark detections (median of ~10 samples) to robustly estimate pitch.

## Topics
- Input
  - `~image` (`sensor_msgs/msg/Image`): subscribe via `image_transport` (raw recommended)
  - `~camera_info` (`sensor_msgs/msg/CameraInfo`): intrinsics for calibration
- Output
  - `~image_with_lines` (`sensor_msgs/msg/Image`): debug image with detected lines overlaid (BGR8)
  - `~lines` (`std_msgs/msg/Float32MultiArray`): detected line segments, each as `[x1, y1, x2, y2]` in a flat array
  - `~markers` (`visualization_msgs/msg/MarkerArray`): optional RViz markers for lines (controlled by a parameter)

## QoS and Latency Optimization
- Subscription QoS: `rclcpp::SensorDataQoS().keep_last(1).best_effort().durability_volatile()`
  - DDS history depth = 1 ensures the queue always retains only the newest frame.
  - The frame currently being processed is finished (never aborted/dropped mid-processing).
- Use `image_transport` with `TransportHints("raw", tcpNoDelay=true)`.

## Parameters
- I/O
  - `image_topic` (string; default: `image`): subscription topic name (also remappable)
  - `use_color_output` (bool; default: `true`): publish output image as BGR8
- Pre-processing
  - `grayscale` (bool; default: `true`)
  - `blur_ksize` (int; default: `5`) odd values only
  - `blur_sigma` (double; default: `1.5`)
  - `roi` (int[4]; default: `[-1, -1, -1, -1]`) `[x, y, w, h]` (disabled when -1)
  - `downscale` (double; default: `1.0`) `> 0`; 1.0 disables downscaling
- Canny
  - `canny_low` (int; default: `40`), `canny_high` (int; default: `120`)
  - `canny_aperture` (int; default: `3`) allowed: 3, 5, 7
  - `canny_L2gradient` (bool; default: `false`)
- Hough
  - `hough_type` (string; `probabilistic` | `standard`; default: `probabilistic`)
  - Common: `rho` (double; `1.0`), `theta_deg` (double; `1.0`), `threshold` (int; `50`)
  - probabilistic: `min_line_length` (double; `30`), `max_line_gap` (double; `10`)
  - standard: `min_theta_deg` (double; `0`), `max_theta_deg` (double; `180`)
- Visualization
  - `draw_color_bgr` (int[3]; default: `[0, 255, 0]`), `draw_thickness` (int; default: `2`)
  - `publish_markers` (bool; default: `true`)
  - `publish_image_with_lines` (bool; default: `false`): publish the overlay image.
    When `false`, the node does not create the image publisher and skips all
    visualization rendering to reduce CPU load.

## HSV Mask (optional)
- `use_hsv_mask` (bool; default: `true`): enable HSV masking after Canny to isolate the black center line.
- Thresholds:
  - `hsv_lower_h` (int; default: `0`), `hsv_lower_s` (int; default: `0`), `hsv_lower_v` (int; default: `0`)
  - `hsv_upper_h` (int; default: `180`), `hsv_upper_s` (int; default: `120`), `hsv_upper_v` (int; default: `150`)
- Morphology:
  - `hsv_dilate_kernel` (int; default: `3`): odd kernel size.
  - `hsv_dilate_iter` (int; default: `1`): dilation iterations (0 to disable).

## Edge Closing (optional)
- `use_edge_close` (bool; default: `true`): apply morphological closing on edges to reduce fragmentation.
- `edge_close_kernel` (int; default: `3`): odd kernel size for rectangular structuring element.
- `edge_close_iter` (int; default: `1`): number of closing iterations (0 to disable).

## Temporal Smoothing (optional)
- `enable_temporal_smoothing` (bool; default: `true`): publish temporally smoothed line segments.
- `ema_alpha` (double; default: `0.5`): EMA factor for endpoints; closer to 1.0 trusts the current frame more.
- `match_max_px` (int; default: `20`): maximum sum of endpoint distances (with best endpoint pairing) to consider a match.
- `match_max_angle_deg` (double; default: `10.0`): maximum angle difference between track and detection.
- `min_age_to_publish` (int; default: `2`): minimum matched frames before a track is published.
- `max_missed` (int; default: `3`): maximum consecutive frames a track can miss before removal.

Algorithm: per frame, greedily match detections to existing tracks using endpoint distance and angle thresholds. Update matched tracks via EMA on endpoints; create new tracks for unmatched detections; age and prune stale tracks. Publish tracks whose age ≥ `min_age_to_publish` (fallback to raw detections if no stable tracks exist in a frame).

## Processing Flow
1. Convert the received image to `cv::Mat` via `cv_bridge`.
2. Apply ROI cropping → downscale (`downscale`).
3. Convert to grayscale if needed → apply GaussianBlur.
4. Run Canny edge detection.
   - If enabled, apply HSV mask to the Canny edges, then optional edge closing (morphological close) before Hough.
5. Run Hough transform
6. If temporal smoothing is enabled, match detections to existing tracks and publish smoothed segments (otherwise publish raw detections).
   - `probabilistic`: use `cv::HoughLinesP` to obtain segments (x1, y1, x2, y2)
   - `standard`: use `cv::HoughLines` to obtain (rho, theta), then extend to image borders to form segments
6. Invert ROI/downscale to restore coordinates to the original scale.
7. Draw overlays → publish `image_with_lines`, and publish `lines` and `markers`.

### Calibration math (summary)
- Let `v` be the median image row of the gray circle center (pixels), `fy, cy` from camera intrinsics, `u = (v - cy)/fy`.
- Let `h` be camera height [m], `D` the known forward ground distance to the circle [m].
- From a pinhole model with pitch `θ` (downward positive), ray–ground intersection yields:
  - `tan(θ) = (D * u - h) / (D + h * u)` and `θ = atan(tan(θ))`.
  - Clamp to a plausible range (±45 deg) and transition to Ready.

## Timing Logs (steady_clock)
- Capture `std::chrono::steady_clock::now()` at the start of the image callback.
- Capture again just before exit, compute the delta in ms, and log at INFO.
- Example: `Processed frame: 12 lines in 6.4 ms`.
- Optionally add DEBUG logs for each stage (Canny/Hough).

## Dynamic Parameter Updates
- Use `on_set_parameters_callback` to update key parameters for pre-processing, Canny/Hough, and visualization at runtime.
- QoS and topic name changes require re-subscription; reject such runtime changes (restart instead).

## Error Handling
- Invalid ROI is disabled with a WARN log.
- Unsupported encodings are converted to BGR/mono if possible; otherwise log WARN and skip the frame.
- Coerce `downscale <= 0` to `1.0`.
- Reject out-of-range parameters for Canny/Hough at update time.

## Dependencies
- `rclcpp`, `sensor_msgs`, `std_msgs`, `visualization_msgs`
- `image_transport`, `cv_bridge`
- OpenCV: `core`, `imgproc`, `imgcodecs`

## File Layout (planned)
- Package: `etrobo_line_detector`
  - `CMakeLists.txt`, `package.xml`
  - `src/etrobo_line_detector.cpp` (node class and `main()` in the same file)
  - `launch/etrobo_line_detector.launch.py`
  - `config/params.yaml`

## Launch Examples
- Normal launch:
  - `ros2 run etrobo_line_detector etrobo_line_detector`
- Parameter tuning example:
  - `ros2 run etrobo_line_detector etrobo_line_detector --ros-args -p image_topic:=/camera/image_raw -p canny_low:=80 -p canny_high:=200 -p hough_type:=probabilistic`
