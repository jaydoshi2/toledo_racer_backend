from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://neondb_owner:q4LlY8NkwmEP@ep-summer-unit-a5earqnh-pooler.us-east-2.aws.neon.tech/sem4_Gp_Project?sslmode=require"

# Create the SQLAlchemy engine with pool settings for serverless compatibility
engine = create_engine(
    DATABASE_URL,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True  # Verify connection health before use
)

# Set up a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()