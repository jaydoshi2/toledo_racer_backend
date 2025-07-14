import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()
bridge = CvBridge()
frame = None

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

    def camera_callback(self, msg):
        global frame
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info("Received frame")  # Log when a frame is received

def generate_frames():
    global frame
    while True:
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print("No frame available")  # Debug log
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('placeholder.jpg', 'rb').read() + b'\r\n')
        asyncio.sleep(0.033)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def main():
    rclpy.init()
    node = CameraSubscriber()
    import threading
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
