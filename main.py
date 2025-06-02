from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import asyncio
import socketio
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base
from schemas import UserCreate, User, ModelCreate, Model
from crud import create_user, get_user, get_models, get_model, create_model
from auth import authenticate_user, create_access_token, get_current_user
from datetime import timedelta
 
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



app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API 5: Login
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# API 5: Signin (User Registration)
@app.post("/users/", response_model=User)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return create_user(db=db, user=user)

# API 1: Store Model Parameters
@app.post("/models/", response_model=Model)
def create_new_model(model: ModelCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return create_model(db=db, model=model, user_id=current_user.id)

# API 2: Train Model (Placeholder with Epoch Response)
@app.post("/train/{model_id}")
async def train_model(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    model = get_model(db, model_id)
    if not model or model.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Model not found or not authorized")
    # Placeholder: Simulate training and send epoch results
    return {"message": f"Training started for model {model_id}", "epochs": model.epochs}

# API 3: ROS Video (Placeholder)
@app.get("/ros/video")
async def get_ros_video():
    return {"message": "ROS video streaming not implemented yet"}

# API 4: Retrieve Data of a Specific Model
@app.get("/models/{model_id}", response_model=Model)
def read_model(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    model = get_model(db, model_id)
    if not model or model.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Model not found or not authorized")
    return model

# API 6: Retrieve List of Models for a User
@app.get("/models/", response_model=list[Model])
def read_models(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_models(db, user_id=current_user.id)

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


# Run the server
if __name__ == "__main__":  # Fix: __name__ not _name_
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)