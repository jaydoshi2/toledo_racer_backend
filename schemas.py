from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str
    email: str

class User(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        orm_mode = True

class ModelCreate(BaseModel):
    title: str
    description: str
    code: str
    epochs: int

class Model(BaseModel):
    id: int
    title: str
    description: str
    code: str
    epochs: int
    user_id: int

    class Config:
        orm_mode = True