# backend_ws.py
import threading
import rclpy
from rclpy.node import Node
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from std_msgs.msg import String
import uvicorn

app = FastAPI()

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ROS2 spinning in background
def ros_spin():
    rclpy.spin(publisher_node)

threading.Thread(target=ros_spin, daemon=True).start()

# WebSocket endpoint
@app.websocket("/ws/control")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received command: {data}")
            publisher_node.publish_command(data)
    except Exception as e:
        print(f"WebSocket closed: {e}")
