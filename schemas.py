from pydantic import BaseModel
from typing import List, Optional
from models import TrainingStatus


# User Schemas
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int

    class Config:
        from_attributes = True


# Drone Details Sub-schema (matches droneDetails JSON from localStorage)
class DroneDetails(BaseModel):
    raceType: str
    algorithm: str
    flightAltitude: str
    velocityLimit: str
    yawLimit: str
    enableWind: str
    windSpeed: str


# Drone Model Schemas
class DroneModelBase(BaseModel):
    title: str           # Maps to modelName from localStorage
    description: str
    training_epochs: int
    drone_details: DroneDetails  # New field to store JSON structure from localStorage


class DroneModelCreate(DroneModelBase):
    pass


class DroneModelUpdate(BaseModel):
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    status: Optional[TrainingStatus] = None


class DroneModel(DroneModelBase):
    id: int
    user_id: int
    drone_id: str
    status: TrainingStatus
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None

    class Config:
        from_attributes = True


class DroneModelList(BaseModel):
    models: List[DroneModel]
