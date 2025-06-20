# filename: drone_pid_controller.py

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Vector3
from px4_msgs.msg import VehicleOdometry, VehicleRatesSetpoint, VehicleThrustSetpoint

class PIDDroneController(Node):
    def __init__(self):
        super().__init__('pid_drone_controller')

        # Constants
        self.origin = np.array([0.0, 0.0])
        self.goal = np.array([10.0, 10.0])  # Arbitrary goal point
        self.path_vector = self.goal - self.origin
        self.path_unit_vector = self.path_vector / np.linalg.norm(self.path_vector)

        # PID parameters (P-control only for now)
        self.Kp_position = 1.0
        self.Kp_angle = 1.0

        # Subscribers
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odometry_callback, 10)

        # Publishers
        self.rate_pub = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', 10)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', 10)

    def odometry_callback(self, msg: VehicleOdometry):
        pos = np.array([msg.position[0], msg.position[1]])  # x, y position
        angle = msg.q  # Quaternion [w,x,y,z]
        orientation_vector = self.quaternion_to_heading(angle)

        # 1. Calculate perpendicular error to path
        error_vector = pos - self.origin
        projection = np.dot(error_vector, self.path_unit_vector)
        closest_point_on_path = self.origin + projection * self.path_unit_vector
        lateral_error_vector = pos - closest_point_on_path
        lateral_error = np.linalg.norm(lateral_error_vector)
        direction = np.cross(self.path_unit_vector, lateral_error_vector)[-1]
        lateral_error *= np.sign(direction)  # Left = -, Right = +

        # 2. Angle error
        path_direction_angle = np.arctan2(self.path_vector[1], self.path_vector[0])
        angle_error = path_direction_angle - np.arctan2(orientation_vector[1], orientation_vector[0])
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize to [-pi, pi]

        # 3. P-Control based on errors
        yaw_rate_cmd = self.Kp_angle * angle_error
        lateral_cmd = self.Kp_position * lateral_error

        self.send_rate_command(yaw_rate_cmd)
        self.send_thrust_command(0.6)  # keep a constant thrust or vary if needed

        self.get_logger().info(f"LatErr: {lateral_error:.2f}, AngErr: {angle_error:.2f}, YawRateCmd: {yaw_rate_cmd:.2f}")

    def quaternion_to_heading(self, q):
        # Converts quaternion to 2D heading vector
        w, x, y, z = q
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return np.array([np.cos(yaw), np.sin(yaw)])

    def send_rate_command(self, yaw_rate):
        msg = VehicleRatesSetpoint()
        msg.roll = 0.0
        msg.pitch = 0.0
        msg.yaw = float(yaw_rate)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.rate_pub.publish(msg)

    def send_thrust_command(self, thrust_val):
        msg = VehicleThrustSetpoint()
        msg.xyz = [0.0, 0.0, -thrust_val]  # Z is negative in PX4 thrust
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.thrust_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PIDDroneController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 