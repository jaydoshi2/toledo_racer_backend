# backend.py
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

rclpy.init()

class DronePublisher(Node):
    def __init__(self):
        super().__init__('drone_command_publisher')
        self.publisher_ = self.create_publisher(String, '/control_commands', 10)

    def publish_command(self, command: str):
        msg = String()
        msg.data = command
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')

publisher_node = DronePublisher()

def ros_spin():
    rclpy.spin(publisher_node)

threading.Thread(target=ros_spin, daemon=True).start()

class CommandRequest(BaseModel):
    command: str

@app.post("/send-command/")
def send_command(request: CommandRequest):
    publisher_node.publish_command(request.command)
    return {"status": "sent", "command": request.command}
