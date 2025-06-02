from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

# Replace with your Neon DB connection string
DATABASE_URL = "postgresql://user:password@host/dbname?sslmode=require"

# Create the SQLAlchemy engine with pool settings for serverless compatibility
engine = create_engine(
    DATABASE_URL,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True  # Verify connection health before use
)

# Set up a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define a base for your models
Base = declarative_base()

# Example model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)

# Create tables in the database (if they don’t exist)
Base.metadata.create_all(bind=engine)

# Example function to query users
def get_users(db: SessionLocal):
    return db.query(User).all()

# Example usage
with SessionLocal() as db:
    users = get_users(db)
    for user in users:
        print(f"User: {user.username}, Email: {user.email}")