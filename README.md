# etrobo_runner

etrobo_runner is a ROS2 node to drive a two-wheeled robot using a PD controller based on luminance input from a
r sensor.


## Features
- Subscribes to `/color` topic (ColorRGBA) for luminance data.
- Publishes velocity commands as `geometry_msgs/msg/Twist` messages on `/cmd_vel`.
- Uses PD control on luminance error to steer the robot along a line.
- Configurable parameters for luminance threshold, proportional gain (kp), and derivative gain (kd).
- Constant forward linear velocity.

## Parameters
- `threshold` (float): Luminance threshold for the controller, default 0.46.
- `kp` (float): Proportional gain, default -0.64.
- `kd` (float): Derivative gain, default -1.2.

## Usage
Run the node using ROS2 launch or directly with:

 run etrobo_runner etrobo_runner

The node will automatically subscribe to color sensor data and publish commands to drive the robot.

## Line Tracing in Simulator Environment

### Requirements

- Ubuntu 22.04
- ROS2 Humble
- Gazebo 11.10.2

### Build

```bash
mkdir -p ros2_ws/src
cd ros2_ws
git clone https://github.com/owhinata/etrobo_simulator.git src/etrobo_simulator
git clone https://github.com/owhinata/etrobo_runner.git src/etrobo_runner
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Run Gazebo Simulator for L-Course

```bash
ros2 launch etrobo_simulator etrobo_world.launch.py x:=-0.4 y:=1.57 Y:=-1.570796327
```

### Run Runner Node

In a separate terminal:

```bash
run etrobo_runner etrobo_runner
```
