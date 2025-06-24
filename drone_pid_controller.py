import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class DroneCommandSubscriber(Node):

    def __init__(self):
        super().__init__('drone_command_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/control_commands',
            self.listener_callback,
            10
        )
        print("DroneCommandSubscriber initialized")
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        print("The command is sent")

    def listener_callback(self, msg):
        self.get_logger().info(f"Command received: {msg.data}")
        twist = Twist()

        if msg.data == "takeoff":
            self.takeoff()
        elif msg.data == "move_forward":
            twist.linear.x = 0.5
            self.cmd_vel_publisher.publish(twist)
        elif msg.data == "move_backward":
            twist.linear.x = -0.5
            self.cmd_vel_publisher.publish(twist)
        elif msg.data == "turn_left":
            twist.angular.z = 1.0
            self.cmd_vel_publisher.publish(twist)
        elif msg.data == "turn_right":
            twist.angular.z = -1.0
            self.cmd_vel_publisher.publish(twist)
        else:
            self.get_logger().warn(f"Unknown command: {msg.data}")

    def takeoff(self):
        self.get_logger().info("Taking off...")
        twist = Twist()
        twist.linear.z = 1.0  # Upward force

        start_time = self.get_clock().now().to_msg().sec
        duration = 2  # seconds of upward thrust

        end_time = start_time + duration
        while self.get_clock().now().to_msg().sec < end_time:
            self.cmd_vel_publisher.publish(twist)
            time.sleep(0.1)

        self.get_logger().info("Takeoff complete")

def main(args=None):
    rclpy.init(args=args)
    node = DroneCommandSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
