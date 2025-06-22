"""Runner node for a two-wheeled robot."""

from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
import rclpy


class Runner(Node):
    """ROS2 node that drives a two-wheeled robot.

    Parameters
    ----------
    threshold : float
        Luminance threshold for the PD controller.
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.
    """

    def __init__(self) -> None:
        """Initialize subscriptions, publishers and timer."""
        super().__init__('etrobo_runner')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(ColorRGBA, '/color', self.color_callback, 10)

        self.luminance = 0.0
        self.prev_error = 0.0

        self.threshold = self.declare_parameter('threshold', 0.5).value
        self.kp = self.declare_parameter('kp', 1.0).value
        self.kd = self.declare_parameter('kd', 0.0).value

        self.timer_period = 0.01
        self.create_timer(self.timer_period, self.publish_cmd_vel)

    def color_callback(self, msg: ColorRGBA) -> None:
        """Store the received luminance value."""
        self.luminance = float(msg.a)

    def publish_cmd_vel(self) -> None:
        """Publish Twist message based on PD control."""
        error = self.luminance - self.threshold
        derivative = (error - self.prev_error) / self.timer_period
        angular = self.kp * error + self.kd * derivative
        self.prev_error = error

        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = angular
        self.publisher_.publish(twist)


def main() -> None:
    """Entry point for the runner node."""
    rclpy.init()
    runner = Runner()
    rclpy.spin(runner)
    runner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
