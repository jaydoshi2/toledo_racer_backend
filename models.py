from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from database import Base

class TrainingStatus(enum.Enum):
    INITIAL = "initial"
    TRAINING = "training"
    FINISHED = "finished"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    drone_models = relationship("UserDroneModel", back_populates="owner")

class UserDroneModel(Base):
    __tablename__ = "user_drone_models"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    drone_id = Column(String, unique=True, index=True)  # Random drone ID
    title = Column(String)
    description = Column(String)
    training_epochs = Column(Integer)
    status = Column(Enum(TrainingStatus), default=TrainingStatus.INITIAL)
    train_loss = Column(Float, nullable=True)
    train_accuracy = Column(Float, nullable=True)
    owner = relationship("User", back_populates="drone_models")