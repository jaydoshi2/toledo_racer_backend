from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import models, schemas
import uuid
from typing import List, Optional

def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    db_user = models.User(username=user.username)
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError:
        db.rollback()
        raise ValueError("Username already exists")

def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.username == username).first()

def create_drone_model(db: Session, user_id: int, model: schemas.DroneModelCreate) -> models.UserDroneModel:
    drone_id = str(uuid.uuid4())  # Generate random drone ID
    db_model = models.UserDroneModel(
        user_id=user_id,
        drone_id=drone_id,
        title=model.title,
        description=model.description,
        training_epochs=model.training_epochs
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def update_drone_model(db: Session, model_id: int, update_data: schemas.DroneModelUpdate) -> Optional[models.UserDroneModel]:
    db_model = db.query(models.UserDroneModel).filter(models.UserDroneModel.id == model_id).first()
    if not db_model:
        return None
    
    for field, value in update_data.dict(exclude_unset=True).items():
        setattr(db_model, field, value)
    
    db.commit()
    db.refresh(db_model)
    return db_model

def get_user_models(db: Session, user_id: int) -> List[models.UserDroneModel]:
    return db.query(models.UserDroneModel).filter(models.UserDroneModel.user_id == user_id).all()

def get_model_by_id(db: Session, model_id: int) -> Optional[models.UserDroneModel]:
    return db.query(models.UserDroneModel).filter(models.UserDroneModel.id == model_id).first()

def get_model_by_drone_id(db: Session, drone_id: str) -> Optional[models.UserDroneModel]:
    return db.query(models.UserDroneModel).filter(models.UserDroneModel.drone_id == drone_id).first()