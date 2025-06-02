from sqlalchemy.orm import Session
from models import User, Model
from schemas import UserCreate, ModelCreate
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_models(db: Session, user_id: int):
    return db.query(Model).filter(Model.user_id == user_id).all()

def get_model(db: Session, model_id: int):
    return db.query(Model).filter(Model.id == model_id).first()

def create_model(db: Session, model: ModelCreate, user_id: int):
    db_model = Model(**model.dict(), user_id=user_id)
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model