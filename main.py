from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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

# Create database tables
models.Base.metadata.create_all(bind=engine)
app = FastAPI()


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
