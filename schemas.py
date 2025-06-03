from pydantic import BaseModel
from typing import List, Optional
from models import TrainingStatus

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int

    class Config:
        from_attributes = True

class DroneModelBase(BaseModel):
    title: str
    description: str
    training_epochs: int

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