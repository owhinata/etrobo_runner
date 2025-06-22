## Design Overview
- **Vehicle control**: publish to `/cmd_vel` at 10 ms intervals.  
- **Steering algorithm**: subscribe to `/color` (type: `std_msgs::msg::ColorRGBA`).  
  - `/color` carries data from a downward-facing color sensor beneath the vehicle.  
  - `R`, `G`, and `B` fields contain standard color values (`0.0–1.0`).  
  - The `A` field holds the luminance value (`0.0–1.0`).  
  - If luminance is below a threshold (dark), steer left; if above the threshold (bright), steer right.
  - The steering angle is determined by a PD controller using **sensor value − threshold** as the error.
    The derivative term is computed from the change in error between control intervals.
  - Threshold and gain parameters are configurable via ROS 2 parameters.
  - The vehicle remains stationary until the first `/color` message is received.

