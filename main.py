from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from models import Base
from schemas import UserCreate, User, DroneModelCreate, DroneModel, DroneModelUpdate
from crud import (
    create_user, get_user_by_username, create_drone_model, 
    update_drone_model, get_user_models, get_model_by_id,
    get_model_by_drone_id
)
from crud import get_user_model_by_username_and_drone_id
from typing import List
import threading
import rclpy
from rclpy.node import Node
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from std_msgs.msg import String
import uvicorn
from typing import Dict, Optional
import logging
import threading
import subprocess
import psutil
import os
import time
import json
import cv2
import base64
from pathlib import Path

import time
import signal
import threading
from queue import Queue

import select
import fcntl
# Create database tables
models.Base.metadata.create_all(bind=engine)
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(2, 1)
        
            def forward(self, x):
                return torch.sigmoid(self.fc(x))
def generate_data():
            np.random.seed(42)
            X = np.random.randn(100, 2)
            y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
     allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: Dict[str, WebSocket] = {}
active_processes: Dict[str, subprocess.Popen] = {}


class GazeboManager:
    def __init__(self):
        self.processes = {}  # Store processes per model
        self.px4_path = Path("/home/jay/PX4/PX4-Autopilot")  # Adjust path as needed
        self.qgroundcontrol_path = Path("./QGroundControl-x86_64.AppImage")
        
    # async def start_gazebo(self, model_id: str, username: str):
    #     """Start Gazebo simulation for specific model"""
    #     try:
    #         logger.info(f"Starting Gazebo for model {model_id}, user {username}")
    #
    #         # Check if PX4 directory exists
    #         if not self.px4_path.exists():
    #             print(f"PX4-Autopilot directory not found at {self.px4_path}")
    #             return False, "PX4-Autopilot directory not found"
    #
    #         # Change to PX4 directory
    #         original_cwd = os.getcwd()
    #         os.chdir(self.px4_path)
    #
    #         # Start PX4 SITL with Gazebo
    #         cmd = ["make", "px4_sitl", "gz_x500"]
    #         logger.info(f"Executing command: {' '.join(cmd)}")
    #
    #         process = subprocess.Popen(
    #             cmd,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True,
    #             env=os.environ.copy()
    #         )
    #
    #         # Store process reference
    #         key = f"{username}_{model_id}"
    #         self.processes[key] = process
    #
    #         # Wait for startup
    #         await asyncio.sleep(10)
    #
    #         # Check if process is still running
    #         if process.poll() is None:
    #             logger.info(f"Gazebo started successfully for {key}")
    #
    #             # Start QGroundControl in background
    #             await self.start_qgroundcontrol(model_id, username)
    #
    #             # Return to original directory
    #             os.chdir(original_cwd)
    #             return True, "Gazebo and QGroundControl started successfully"
    #         else:
    #             stdout, stderr = process.communicate()
    #             logger.error(f"Gazebo failed to start: {stderr}")
    #             os.chdir(original_cwd)
    #             return False, f"Gazebo failed to start: {stderr}"
    #
    #     except Exception as e:
    #         logger.error(f"Error starting Gazebo: {e}")
    #         os.chdir(original_cwd)
    #         return False, f"Error starting Gazebo: {str(e)}"
    
    async def start_gazebo(self, model_id: str, username: str):
        """Start Gazebo simulation for specific model"""
        try:
            logger.info(f"Starting Gazebo for model {model_id}, user {username}")
            
            if not self.px4_path.exists():
                return False, "PX4-Autopilot directory not found"
            
            original_cwd = os.getcwd()
            os.chdir(self.px4_path)
            
            # Set environment variables for headless operation
            env = os.environ.copy()
            env['HEADLESS'] = '0'  # Keep GUI for streaming
            env['PX4_SIM_MODEL'] = 'gz_x500'
            
            # Start PX4 SITL with Gazebo
            cmd = ["make", "px4_sitl", "gz_x500"]
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            key = f"{username}_{model_id}"
            self.processes[key] = process
            
            # Wait longer for Gazebo to fully start
            await asyncio.sleep(15)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"Gazebo started successfully for {key}")
                os.chdir(original_cwd)
                return True, "Gazebo started successfully"
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Gazebo failed to start: {stderr}")
                os.chdir(original_cwd)
                return False, f"Gazebo failed to start: {stderr}"
                
        except Exception as e:
            logger.error(f"Error starting Gazebo: {e}")
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            return False, f"Error starting Gazebo: {str(e)}"

    async def start_qgroundcontrol(self, model_id: str, username: str):
        """Start QGroundControl"""
        try:
            if self.qgroundcontrol_path.exists():
                # Make executable
                os.chmod(self.qgroundcontrol_path, 0o755)
                
                # Start QGroundControl
                qgc_process = subprocess.Popen(
                    [str(self.qgroundcontrol_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                key = f"{username}_{model_id}_qgc"
                self.processes[key] = qgc_process
                logger.info(f"QGroundControl started for {username}_{model_id}")
                
        except Exception as e:
            logger.error(f"Error starting QGroundControl: {e}")
    
    def stop_gazebo(self, model_id: str, username: str):
        """Stop Gazebo simulation"""
        key = f"{username}_{model_id}"
        qgc_key = f"{username}_{model_id}_qgc"
        
        # Stop Gazebo
        if key in self.processes:
            process = self.processes[key]
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.processes[key]
            logger.info(f"Stopped Gazebo for {key}")
        
        # Stop QGroundControl
        if qgc_key in self.processes:
            process = self.processes[qgc_key]
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.processes[qgc_key]
            logger.info(f"Stopped QGroundControl for {qgc_key}")
    
    def get_status(self, model_id: str, username: str) -> str:
        """Get simulation status"""
        key = f"{username}_{model_id}"
        if key in self.processes:
            process = self.processes[key]
            if process.poll() is None:
                return "training"
            else:
                return "failed"
        return "idle"
    def capture_gazebo_window(window_id):
        """Use ffmpeg to capture the Gazebo window with a specific ID."""
        command = [
            "ffmpeg",
            "-y",
            "-f", "x11grab",
            "-i", f":0+0,0",
            "-window_id", window_id,
            "-video_size", "640x480",
            "-vcodec", "mjpeg",
            "-r", "30",
            "-f", "image2pipe",
            "-"
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return process
    def find_gazebo_window():
        """Find the Gazebo window ID using xdotool."""
        try:
            result = subprocess.check_output(["xdotool", "search", "--name", "Gazebo"]).decode().strip().split('\n')
            return result[0] if result else None
        except Exception as e:
            print(f"Error finding Gazebo window: {e}")
            return None
    def generate_placeholder():
        """Return a placeholder frame when Gazebo is not available."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Gazebo Stream Not Found", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
# Initialize Gazebo Manager
gazebo_manager = GazeboManager()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

@app.post("/users/", response_model=User)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username already exists
    existing_user = get_user_by_username(db, user.username)
    if existing_user:
        print(f"Username {user.username} already exists")
        raise HTTPException(status_code=400, detail="Username already exists")
    try:
        print(f"Creating user {user.username}")
        return create_user(db=db, user=user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Get user by username
@app.get("/users/{username}", response_model=User)
def get_user(username: str, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=username)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.get("/users/{username}/drone-models/", response_model=List[DroneModel])
def get_all_models_for_user(username: str, db: Session = Depends(get_db)):
    user = get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    models_list = get_user_models(db, user.id)
    return models_list

# Create drone model for user
@app.post("/users/{username}/drone-models/", response_model=DroneModel)
def create_drone_model_for_user(username: str, model: DroneModelCreate, db: Session = Depends(get_db)):
    user = get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return create_drone_model(db, user_id=user.id, model=model)


# Update drone model metrics (train_loss, train_accuracy, status)
@app.put("/drone-models/{model_id}/", response_model=DroneModel)
def update_drone_model_metrics(model_id: int, update_data: DroneModelUpdate, db: Session = Depends(get_db)):
    updated_model = update_drone_model(db, model_id, update_data)
    if not updated_model:
        raise HTTPException(status_code=404, detail="Drone model not found")
    return updated_model


# Fetch user drone model by username and drone_id
@app.get("/users/{username}/drone-models/{drone_id}", response_model=DroneModel)
def get_user_drone_model(username: str, drone_id: str, db: Session = Depends(get_db)):
    db_model = get_user_model_by_username_and_drone_id(db, username, drone_id)
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found for given user and drone ID")
    return db_model


@app.websocket("/ws/train")
async def train_model(websocket: WebSocket):
    await websocket.accept()

    try:
        model = SimpleModel()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        X, y = generate_data()
        num_epochs = 20

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y).float().mean().item()

            metrics = {
                "epoch": epoch + 1,
                "total_epochs": num_epochs,
                "loss": round(loss.item(), 4),
                "accuracy": round(accuracy, 4),
                "progress": round((epoch + 1) / num_epochs * 100, 1),
                "done": False,
            }

           # print(f"Epoch {epoch+1}/{num_epochs} | Loss: {metrics['loss']} | Accuracy: {metrics['accuracy']}")

            # ✅ Send the metrics to the frontend
            await websocket.send_json(metrics)
            await asyncio.sleep(0.5)  # simulate training time

        await websocket.send_json({"done": True, "message": "Training completed!"})
        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e), "done": True})
        await websocket.close()

@app.post("/users/{username}/drone-models/{model_id}/start-simulation")
async def start_simulation(username: str, model_id: str):
    """Start Gazebo simulation for specific model"""
    try:
        logger.info(f"Starting simulation for user {username}, model {model_id}")
         
        # Check if simulation is already running
        current_status = gazebo_manager.get_status(model_id, username)
        if current_status == "training":
            return {"status": "already_running", "message": "Simulation already running"}
        
        success, message = await gazebo_manager.start_gazebo(model_id, username)
        
        if success:
            # Notify connected WebSocket clients
            key = f"{username}_{model_id}"
            if key in active_connections:
                await active_connections[key].send_json({
                    "type": "simulation_status",
                    "status": "training",
                    "message": message
                })
            
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": message}
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return {"status": "error", "message": str(e)}
    
@app.post("/users/{username}/drone-models/{model_id}/stop-simulation")
async def stop_simulation(username: str, model_id: str):
    """Stop Gazebo simulation"""
    try:
        gazebo_manager.stop_gazebo(model_id, username)
        
        # Notify connected WebSocket clients
        key = f"{username}_{model_id}"
        if key in active_connections:
            await active_connections[key].send_json({
                "type": "simulation_status",
                "status": "idle",
                "message": "Simulation stopped"
            })
        
        return {"status": "success", "message": "Simulation stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        return {"status": "error", "message": str(e)}  
      




#             if window_id:
#                 logger.info(f"Found Gazebo window: {window_id}")
#                 # Capture specific window
#                 command = [
#                     "ffmpeg",
#                     "-f", "x11grab",
#                     "-video_size", "800x600",
#                     "-framerate", "10",
#                     "-i", f":0.0",
#                     "-filter_complex", f"[0:v]crop=800:600:0:0[out]",
#                     "-map", "[out]",
#                     "-vcodec", "mjpeg",
#                     "-q:v", "8",
#                     "-f", "image2pipe",
#                     "-vframes", "9999999",
#                     "-"
#                 ]
#             else:
#                 logger.warning("No Gazebo window found, using screen capture")
#                 # Screen capture with better region detection
#                 command = [
#                     "ffmpeg",
#                     "-f", "x11grab",
#                     "-video_size", "1024x768",
#                     "-framerate", "10",
#                     "-i", ":0.0+0,0",
#                     "-filter_complex", "[0:v]crop=800:600:100:100[out]",
#                     "-map", "[out]",
#                     "-vcodec", "mjpeg",
#                     "-q:v", "8",
#                     "-f", "image2pipe",
#                     "-vframes", "9999999",
#                     "-"
#                 ]
#
#             logger.info(f"Starting FFmpeg: {' '.join(command)}")
#
#             process = subprocess.Popen(
#                 command,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 bufsize=1024*1024
#             )
#
#             # Give FFmpeg time to start
#             time.sleep(1)
#
#             boundary = b"--frame\r\n"
#             content_type = b"Content-Type: image/jpeg\r\n\r\n"
#
#             buffer = b""
#             jpeg_start = b"\xff\xd8"
#             jpeg_end = b"\xff\xd9"
#
#             while True:
#                 try:
#                     # Check if process is still running
#                     if process.poll() is not None:
#                         logger.warning("FFmpeg process died")
#                         break
#
#                     # Read data
#                     chunk = process.stdout.read(4096)
#                     if not chunk:
#                         time.sleep(0.1)
#                         continue
#
#                     buffer += chunk
#
#                     # Look for complete JPEG frames
#                     while True:
#                         start_idx = buffer.find(jpeg_start)
#                         if start_idx == -1:
#                             break
#
#                         end_idx = buffer.find(jpeg_end, start_idx + 2)
#                         if end_idx == -1:
#                             break
#
#                         # Extract complete JPEG frame
#                         jpeg_frame = buffer[start_idx:end_idx + 2]
#                         buffer = buffer[end_idx + 2:]
#
#                         # Yield frame in MJPEG format
#                         frame_count += 1
#                         yield boundary + content_type + jpeg_frame + b"\r\n"
#
#                         # Limit frame rate
#                         if frame_count % 3 == 0:  # Every 3rd frame
#                             time.sleep(0.1)
#
#                 except Exception as e:
#                     logger.error(f"Error reading frame: {e}")
#                     break
#
#         except Exception as e:
#             logger.error(f"Error in video stream: {e}")
#
#         finally:
#             if process:
#                 try:
#                     process.terminate()
#                     process.wait(timeout=3)
#                 except:
#                     process.kill()
#
#             # Send placeholder when stream ends
#             placeholder = generate_placeholder_frame()
#             if placeholder:
#                 yield placeholder
#
#     return StreamingResponse(
#         generate(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )|

# @app.get("/gazebo-stream/{model_id}")
# async def gazebo_stream(model_id: str, username: str = None):
#     """Stream the Gazebo simulation as MJPEG."""
#     logger.info(f"Starting Gazebo stream for model {model_id}, user {username}")
#
#     def generate():
#         process = None
#         frame_count = 0
#
#         try:
#             # Wait a bit for Gazebo to be ready
#             time.sleep(2)
#
#             # Try multiple approaches to capture Gazebo
#             window_id = find_gazebo_window()
#             if window_id:
#                 logger.info(f"Found Gazebo window: {window_id}")
#                 command = [
#                     "ffmpeg",
#                     "-f", "x11grab",
#                     "-video_size", "800x600",
#                     "-framerate", "10",
#                     "-i", ":0.0",
#                     "-filter_complex", "[0:v]crop=800:600:0:0[out]",  # Fixed syntax
#                     "-map", "[out]",
#                     "-vcodec", "mjpeg",
#                     "-q:v", "8",
#                     "-f", "image2pipe",
#                     "-vframes", "9999999",
#                     "-"
#                 ]
#             else:
#                 logger.warning("No Gazebo window found, using screen capture")
#                 command = [
#                     "ffmpeg",
#                     "-f", "x11grab",
#                     "-video_size", "1024x768",
#                     "-framerate", "10",
#                     "-i", ":0.0+0,0",
#                     "-filter_complex", "[0:v]crop=800:600:100:100[out]",  # Fixed syntax
#                     "-map", "[out]",
#                     "-vcodec", "mjpeg",
#                     "-q:v", "8",
#                     "-f", "image2pipe",
#                     "-vframes", "9999999",
#                     "-"
#                 ]
#
#             logger.info(f"Starting FFmpeg: {' '.join(command)}")
#
#             process = subprocess.Popen(
#                 command,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 bufsize=1024*1024
#             )
#
#             # Give FFmpeg time to start
#             time.sleep(1)
#
#             boundary = b"--frame\r\n"
#             content_type = b"Content-Type: image/jpeg\r\n\r\n"
#
#             buffer = b""
#             jpeg_start = b"\xff\xd8"
#             jpeg_end = b"\xff\xd9"
#
#             while True:
#                 try:
#                     # Check if process is still running
#                     if process.poll() is not None:
#                         logger.warning("FFmpeg process died")
#                         break
#
#                     # Read data with timeout
#                     chunk = process.stdout.read(4096)
#                     if not chunk:
#                         time.sleep(0.1)
#                         continue
#
#                     buffer += chunk
#
#                     # Look for complete JPEG frames
#                     while True:
#                         start_idx = buffer.find(jpeg_start)
#                         if start_idx == -1:
#                             break
#
#                         end_idx = buffer.find(jpeg_end, start_idx + 2)
#                         if end_idx == -1:
#                             break
#
#                         # Extract complete JPEG frame
#                         jpeg_frame = buffer[start_idx:end_idx + 2]
#                         buffer = buffer[end_idx + 2:]
#
#                         # Yield frame in MJPEG format
#                         frame_count += 1
#                         yield boundary + content_type + jpeg_frame + b"\r\n"
#
#                         # Limit frame rate
#                         if frame_count % 3 == 0:  # Every 3rd frame
#                             time.sleep(0.1)
#
#                 except Exception as e:
#                     logger.error(f"Error reading frame: {e}")
#                     break
#
#         except GeneratorExit:
#             logger.info("Generator exit requested")
#             raise
#         except Exception as e:
#             logger.error(f"Error in video stream: {e}")
#
#         finally:
#             # Cleanup process
#             if process:
#                 try:
#                     process.terminate()
#                     process.wait(timeout=3)
#                 except subprocess.TimeoutExpired:
#                     process.kill()
#                 except Exception as e:
#                     logger.error(f"Error cleaning up process: {e}")
#
#     return StreamingResponse(
#         generate(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )
#
@app.get("/gazebo-stream/{model_id}")
async def gazebo_stream(model_id: str, username: str = None):
    """Stream the Gazebo simulation as MJPEG."""
    logger.info(f"Starting Gazebo stream for model {model_id}, user {username}")
    
    def generate():
        process = None
        frame_count = 0
        
        try:
            time.sleep(2)
            window_id = find_gazebo_window()
            
            if window_id:
                logger.info(f"Found Gazebo window: {window_id}")
                
                # Get window geometry first
                try:
                    geometry = subprocess.check_output([
                        "xwininfo", "-id", window_id
                    ], stderr=subprocess.DEVNULL).decode()
                    
                    # Parse window position and size
                    x_pos = y_pos = width = height = 0
                    for line in geometry.split('\n'):
                        if 'Absolute upper-left X:' in line:
                            x_pos = int(line.split(':')[1].strip())
                        elif 'Absolute upper-left Y:' in line:
                            y_pos = int(line.split(':')[1].strip())
                        elif 'Width:' in line:
                            width = int(line.split(':')[1].strip())
                        elif 'Height:' in line:
                            height = int(line.split(':')[1].strip())
                    
                    logger.info(f"Window geometry: {width}x{height} at {x_pos},{y_pos}")
                    
                    # Capture the specific window area
                    command = [
                        "ffmpeg",
                        "-f", "x11grab",
                        "-video_size", f"{width}x{height}",
                        "-framerate", "10",
                        "-i", f":0.0+{x_pos},{y_pos}",  # Capture specific window position
                        "-vcodec", "mjpeg",
                        "-q:v", "8",
                        "-f", "image2pipe",
                        "-vframes", "9999999",
                        "-"
                    ]
                    
                except Exception as e:
                    logger.error(f"Error getting window geometry: {e}")
                    # Fallback to window ID capture
                    command = [
                        "ffmpeg",
                        "-f", "x11grab",
                        "-video_size", "800x600",
                        "-framerate", "10",
                        "-i", f":0.0",
                        "-filter_complex", f"[0:v]crop=iw:ih:0:0[out]",
                        "-map", "[out]",
                        "-vcodec", "mjpeg",
                        "-q:v", "8",
                        "-f", "image2pipe",
                        "-vframes", "9999999",
                        "-"
                    ]
            else:
                logger.warning("No Gazebo window found, using full screen capture")
                command = [
                    "ffmpeg",
                    "-f", "x11grab",
                    "-video_size", "1920x1080",  # Use full screen size
                    "-framerate", "5",
                    "-i", ":0.0",
                    "-vcodec", "mjpeg",
                    "-q:v", "8",
                    "-f", "image2pipe",
                    "-vframes", "9999999",
                    "-"
                ]
            
            logger.info(f"Starting FFmpeg: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024*1024
            )
            
            time.sleep(1)
            
            boundary = b"--frame\r\n"
            content_type = b"Content-Type: image/jpeg\r\n\r\n"
            
            buffer = b""
            jpeg_start = b"\xff\xd8"
            jpeg_end = b"\xff\xd9"
            
            while True:
                try:
                    if process.poll() is not None:
                        logger.warning("FFmpeg process died")
                        break
                    
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        time.sleep(0.1)
                        continue
                    
                    buffer += chunk
                    
                    while True:
                        start_idx = buffer.find(jpeg_start)
                        if start_idx == -1:
                            break
                            
                        end_idx = buffer.find(jpeg_end, start_idx + 2)
                        if end_idx == -1:
                            break
                            
                        jpeg_frame = buffer[start_idx:end_idx + 2]
                        buffer = buffer[end_idx + 2:]
                        
                        frame_count += 1
                        yield boundary + content_type + jpeg_frame + b"\r\n"
                        
                        if frame_count % 2 == 0:  # Reduce frame rate
                            time.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Error reading frame: {e}")
                    break
                    
        except GeneratorExit:
            logger.info("Generator exit requested")
            raise
        except Exception as e:
            logger.error(f"Error in video stream: {e}")
            
        finally:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up process: {e}")
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def find_gazebo_window():
    """Find the Gazebo window ID using multiple methods."""
    try:
        # Method 1: Try xdotool
        try:
            window_names = ["Gazebo", "gazebo", "Ignition Gazebo", "gz sim", "Gazebo*"]
            
            for name in window_names:
                try:
                    result = subprocess.check_output(
                        ["xdotool", "search", "--name", name],
                        stderr=subprocess.DEVNULL,
                        timeout=5
                    ).decode().strip()
                    
                    if result:
                        window_ids = result.split('\n')
                        if window_ids and window_ids[0]:
                            logger.info(f"Found window with xdotool: {window_ids[0]}")
                            return window_ids[0]
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue
        except:
            pass
            
        # Method 2: Try wmctrl
        try:
            result = subprocess.check_output(
                ["wmctrl", "-l"],
                stderr=subprocess.DEVNULL,
                timeout=5
            ).decode()
            
            for line in result.split('\n'):
                if 'gazebo' in line.lower() or 'Gazebo' in line:
                    window_id = line.split()[0]
                    logger.info(f"Found window with wmctrl: {window_id}")
                    return window_id
        except:
            pass
            
        # Method 3: Try xwininfo
        try:
            result = subprocess.check_output(
                ["xwininfo", "-root", "-tree"],
                stderr=subprocess.DEVNULL,
                timeout=5
            ).decode()
            
            for line in result.split('\n'):
                if 'gazebo' in line.lower() or 'Gazebo' in line:
                    parts = line.strip().split()
                    if parts and parts[0].startswith('0x'):
                        window_id = parts[0]
                        logger.info(f"Found window with xwininfo: {window_id}")
                        return window_id
        except:
            pass
            
        logger.warning("No Gazebo window found with any method")
        return None
        
    except Exception as e:
        logger.error(f"Error finding Gazebo window: {e}")
        return None

def generate_placeholder_frame():
    """Generate a placeholder frame when stream is not available."""
    try:
        # Create placeholder image
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        frame.fill(50)  # Dark gray background
        
        # Add text
        cv2.putText(frame, "Gazebo Stream", (250, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, "Initializing...", (300, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(frame, "Please wait for Gazebo to start", (200, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        return (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                
    except Exception as e:
        logger.error(f"Error generating placeholder: {e}")
        return None

def get_window_info(window_id):
    """Get detailed window information including position and size."""
    try:
        # Get window info
        info = subprocess.check_output([
            "xwininfo", "-id", window_id
        ], stderr=subprocess.DEVNULL).decode()
        
        # Parse the output
        window_info = {}
        for line in info.split('\n'):
            if 'Absolute upper-left X:' in line:
                window_info['x'] = int(line.split(':')[1].strip())
            elif 'Absolute upper-left Y:' in line:
                window_info['y'] = int(line.split(':')[1].strip())
            elif 'Width:' in line:
                window_info['width'] = int(line.split(':')[1].strip())
            elif 'Height:' in line:
                window_info['height'] = int(line.split(':')[1].strip())
        
        logger.info(f"Window info: {window_info}")
        return window_info
        
    except Exception as e:
        logger.error(f"Error getting window info: {e}")
        return None

# Add this endpoint to test window detection
@app.get("/window-info/{model_id}")
async def get_window_info_endpoint(model_id: str):
    """Get information about the Gazebo window."""
    window_id = find_gazebo_window()
    if window_id:
        info = get_window_info(window_id)
        return {"window_id": window_id, "info": info}
    return {"error": "No Gazebo window found"}

def get_window_info(window_id):
    """Get detailed window information including position and size."""
    try:
        # Get window info
        info = subprocess.check_output([
            "xwininfo", "-id", window_id
        ], stderr=subprocess.DEVNULL).decode()
        
        # Parse the output
        window_info = {}
        for line in info.split('\n'):
            if 'Absolute upper-left X:' in line:
                window_info['x'] = int(line.split(':')[1].strip())
            elif 'Absolute upper-left Y:' in line:
                window_info['y'] = int(line.split(':')[1].strip())
            elif 'Width:' in line:
                window_info['width'] = int(line.split(':')[1].strip())
            elif 'Height:' in line:
                window_info['height'] = int(line.split(':')[1].strip())
        
        logger.info(f"Window info: {window_info}")
        return window_info
        
    except Exception as e:
        logger.error(f"Error getting window info: {e}")
        return None
@app.get("/debug-stream/{model_id}")
async def debug_stream(model_id: str):
    """Debug stream to see what's being captured."""
    
    def generate():
        process = None
        try:
            # Simple full screen capture first
            command = [
                "ffmpeg",
                "-f", "x11grab",
                "-video_size", "1920x1080",
                "-framerate", "2",
                "-i", ":0.0",
                "-vcodec", "mjpeg",
                "-q:v", "5",
                "-f", "mjpeg",
                "pipe:1"
            ]
            
            logger.info(f"Debug FFmpeg: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                yield chunk
                
        except GeneratorExit:
            logger.info("Debug stream generator exit")
            raise
        except Exception as e:
            logger.error(f"Debug stream error: {e}")
        finally:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    process.kill()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
# Add this alternative endpoint for testing
@app.get("/test-stream/{model_id}")
async def test_stream(model_id: str):
    """Test stream with simple screen capture."""
    
    def generate():
        try:
            # Simple screen capture command
            command = [
                "ffmpeg",
                "-f", "x11grab",
                "-video_size", "800x600",
                "-framerate", "5",
                "-i", ":0.0",
                "-vcodec", "mjpeg",
                "-q:v", "10",
                "-f", "mjpeg",
                "-"
            ]
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                yield chunk
                
        except Exception as e:
            logger.error(f"Test stream error: {e}")
            
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
