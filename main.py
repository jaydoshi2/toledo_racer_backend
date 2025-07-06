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
        self.px4_path = Path("PX4/PX4-Autopilot")  # Adjust path as needed
        self.qgroundcontrol_path = Path("./QGroundControl-x86_64.AppImage")
        
    async def start_gazebo(self, model_id: str, username: str):
        """Start Gazebo simulation for specific model"""
        try:
            logger.info(f"Starting Gazebo for model {model_id}, user {username}")
            
            # Check if PX4 directory exists
            if not self.px4_path.exists():
                return False, "PX4-Autopilot directory not found"
            
            # Change to PX4 directory
            original_cwd = os.getcwd()
            os.chdir(self.px4_path)
            
            # Start PX4 SITL with Gazebo
            cmd = ["make", "px4_sitl", "gz_x500"]
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            # Store process reference
            key = f"{username}_{model_id}"
            self.processes[key] = process
            
            # Wait for startup
            await asyncio.sleep(10)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"Gazebo started successfully for {key}")
                
                # Start QGroundControl in background
                await self.start_qgroundcontrol(model_id, username)
                
                # Return to original directory
                os.chdir(original_cwd)
                return True, "Gazebo and QGroundControl started successfully"
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Gazebo failed to start: {stderr}")
                os.chdir(original_cwd)
                return False, f"Gazebo failed to start: {stderr}"
                
        except Exception as e:
            logger.error(f"Error starting Gazebo: {e}")
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

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {metrics['loss']} | Accuracy: {metrics['accuracy']}")

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
@app.get("/gazebo-stream/{model_id}")
async def gazebo_stream_endpoint(model_id: str, username: str = "default"):
    """Stream Gazebo simulation view"""
    # This is a placeholder - you'll need to implement actual Gazebo streaming
    # You can use OpenCV to capture the Gazebo window or use gstreamer
    
    def generate_stream():
        while True:
            # Placeholder for actual Gazebo capture
            # You would capture the Gazebo window here
            frame = create_placeholder_frame(f"Gazebo Stream for Model {model_id}")
            
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

def create_placeholder_frame(text: str):
    """Create a placeholder frame with text"""
    import numpy as np
    
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (50, 240), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Gazebo simulation would appear here", (50, 280), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    
    return frame
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
