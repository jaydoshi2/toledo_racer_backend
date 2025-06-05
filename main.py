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
from typing import List

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

# Create new drone model
@app.post("/users/{username}/models/", response_model=DroneModel)
def create_new_drone_model(username: str, model: DroneModelCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=username)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return create_drone_model(db=db, user_id=db_user.id, model=model)

# Update model status and metrics
@app.put("/models/{model_id}", response_model=DroneModel)
def update_model(model_id: int, update_data: DroneModelUpdate, db: Session = Depends(get_db)):
    db_model = update_drone_model(db, model_id=model_id, update_data=update_data)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model

# Get all models for a user
@app.get("/users/{username}/models/", response_model=List[DroneModel])
def get_models(username: str, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=username)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return get_user_models(db=db, user_id=db_user.id)

# Get model by drone_id
@app.get("/models/drone/{drone_id}", response_model=DroneModel)
def get_model_by_drone(drone_id: str, db: Session = Depends(get_db)):
    print(f"Getting model by drone_id {drone_id}")
    db_model = get_model_by_drone_id(db, drone_id=drone_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model

# WebSocket endpoint for training updates
@app.websocket("/ws/train/{model_id}")
async def train_model(websocket: WebSocket, model_id: int, db: Session = Depends(get_db)):
    print(f"Training model {model_id}")
    await websocket.accept()
    
    try:
        # Get model from database
        db_model = get_model_by_id(db, model_id)
        if not db_model:
            await websocket.send_json({"error": "Model not found"})
            await websocket.close()
            return

        # Update status to training
        update_data = DroneModelUpdate(status="training")
        update_drone_model(db, model_id, update_data)

        # Simulate training
        for epoch in range(db_model.training_epochs):
            # Simulate training metrics
            loss = 1.0 - (epoch / db_model.training_epochs)  # Simulated decreasing loss
            accuracy = epoch / db_model.training_epochs  # Simulated increasing accuracy
            
            metrics = {
                "epoch": epoch + 1,
                "total_epochs": db_model.training_epochs,
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "progress": round((epoch + 1) / db_model.training_epochs * 100, 1),
                "done": False
            }
            
            await websocket.send_json(metrics)
            await asyncio.sleep(0.5)  # Simulate training time

        # Update final status and metrics
        final_update = DroneModelUpdate(
            status="finished",
            train_loss=0.1,  # Final simulated loss
            train_accuracy=0.95  # Final simulated accuracy
        )
        update_drone_model(db, model_id, final_update)
        
        await websocket.send_json({
            "done": True,
            "message": "Training completed!",
            "final_loss": 0.1,
            "final_accuracy": 0.95
        })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)
        await websocket.send_json({"error": str(e), "done": True})
    finally:
        await websocket.close()

# @app.websocket("/ws/train")
# async def train_model(websocket: WebSocket):
#     await websocket.accept()

#     try:
#         model = SimpleModel()
#         criterion = nn.BCELoss()
#         optimizer = optim.SGD(model.parameters(), lr=0.1)
#         X, y = generate_data()
#         num_epochs = 20

#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()

#             predictions = (outputs > 0.5).float()
#             accuracy = (predictions == y).float().mean().item()

#             metrics = {
#                 "epoch": epoch + 1,
#                 "total_epochs": num_epochs,
#                 "loss": round(loss.item(), 4),
#                 "accuracy": round(accuracy, 4),
#                 "progress": round((epoch + 1) / num_epochs * 100, 1),
#                 "done": False,
#             }

#             print(f"Epoch {epoch+1}/{num_epochs} | Loss: {metrics['loss']} | Accuracy: {metrics['accuracy']}")

#             # ✅ Send the metrics to the frontend
#             await websocket.send_json(metrics)
#             await asyncio.sleep(0.5)  # simulate training time

#         await websocket.send_json({"done": True, "message": "Training completed!"})
#         await websocket.close()

#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         await websocket.send_json({"error": str(e), "done": True})
#         await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)