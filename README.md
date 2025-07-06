# Toledo Racer Backend

A comprehensive backend system for drone racing simulation and model training, built with FastAPI, ROS2, and MAVSDK. The system consists of two main servers that work together to provide drone control, user management, and machine learning model training capabilities.

## 🏗️ Architecture Overview

The backend is composed of two main servers:

1. **Main Server** (`main.py`) - FastAPI web server handling user management, drone models, and training
2. **Drone Controller Server** (`drone_controller.py`) - ROS2 node for real-time drone control via MAVSDK

## 🚀 Main Server (FastAPI)

**File:** `main.py`  
**Port:** 5000  
**Purpose:** Web API server for user management, drone model operations, and real-time training

### Key Features

#### 🔐 User Management
- **POST** `/users/` - Create new users
- **GET** `/users/{username}` - Get user by username
- **GET** `/users/{username}/drone-models/` - Get all drone models for a user
- **POST** `/users/{username}/drone-models/` - Create drone model for user
- **GET** `/users/{username}/drone-models/{drone_id}` - Get specific drone model

#### 🤖 Drone Model Management
- **PUT** `/drone-models/{model_id}/` - Update drone model metrics and status
- Model training status tracking (initializing, training, finished)
- Training metrics storage (loss, accuracy)

#### 🌐 WebSocket Endpoints
- **WebSocket** `/ws/control` - Real-time drone control commands
- **WebSocket** `/ws/train` - Live model training progress updates

#### 🔧 ROS2 Integration
- Integrated ROS2 publisher for drone commands
- Background ROS2 spinning thread
- Command publishing to `/control_commands` topic

### Database Models

#### User Model
```python
class User:
    id: int (Primary Key)
    username: str (Unique)
    drone_models: Relationship to UserDroneModel
```

#### UserDroneModel
```python
class UserDroneModel:
    id: int (Primary Key)
    user_id: int (Foreign Key)
    drone_id: str (Unique UUID)
    title: str
    description: str
    training_epochs: int
    drone_details: JSON (Race configuration)
    status: str (initializing/training/finished)
    train_loss: float (nullable)
    train_accuracy: float (nullable)
```

### Drone Configuration Schema
```python
class DroneDetails:
    raceType: str
    algorithm: str
    flightAltitude: str
    velocityLimit: str
    yawLimit: str
    enableWind: str
    windSpeed: str
```

## 🎮 Drone Controller Server (ROS2)

**File:** `drone_controller.py`  
**Purpose:** Real-time drone control via MAVSDK and PX4 SITL

### Key Features

#### 🔌 Drone Connection
- Connects to PX4 SITL simulator via UDP (`udp://:14540`)
- Automatic connection state monitoring
- Dedicated async event loop for drone operations

#### 🎯 Supported Commands
- **`takeoff`** - Arm and take off the drone
- **`land`** - Land the drone and stop offboard mode
- **`move_forward`** - Move forward at 2.0 m/s
- **`move_backward`** - Move backward at -2.0 m/s
- **`turn_left`** - Turn left at -30°/s
- **`turn_right`** - Turn right at 30°/s
- **`stop`** - Stop all movement (zero velocity)

#### 🔄 ROS2 Integration
- Subscribes to `/control_commands` topic
- Processes commands asynchronously
- Thread-safe command execution

#### 🛡️ Safety Features
- Automatic offboard mode management
- Connection state validation
- Error handling and logging

## 🗄️ Database Configuration

**File:** `database.py`  
**Type:** PostgreSQL (via SQLAlchemy)

### Connection Settings
- Pool recycling every 30 minutes
- Connection health verification
- Environment variable configuration via `.env`

### Required Environment Variables
```bash
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

## 🔧 Additional Components

### Authentication (`auth.py`)
- JWT token-based authentication
- Password hashing with bcrypt
- OAuth2 password bearer scheme
- Token expiration (30 minutes)

### CRUD Operations (`crud.py`)
- User creation and retrieval
- Drone model management
- Database transaction handling
- Error handling for duplicate entries

### Data Validation (`schemas.py`)
- Pydantic models for request/response validation
- Type safety and data serialization
- Nested schema support for drone details

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- ROS2 (Humble or later)
- PX4 SITL simulator
- PostgreSQL database
- MAVSDK Python library

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd toledo_racer_backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Initialize database**
   ```bash
   python -c "from database import engine; from models import Base; Base.metadata.create_all(bind=engine)"
   ```

### Running the Servers

1. **Start PX4 SITL** (in a separate terminal)
   ```bash
   make px4_sitl gazebo
   ```

2. **Start the Drone Controller** (in a separate terminal)
   ```bash
   python drone_controller.py
   ```

3. **Start the Main Server** (in a separate terminal)
   ```bash
   python main.py
   ```

The main server will be available at `http://localhost:5000`

## 📡 API Documentation

Once the main server is running, you can access:
- **Interactive API docs:** `http://localhost:5000/docs`
- **ReDoc documentation:** `http://localhost:5000/redoc`

## 🔍 Testing

### Connection Testing
Use `debug_connection.py` to test drone connectivity:
```bash
python debug_connection.py
```

### Minimal Test
Use `minimal_test.py` for basic drone operations:
```bash
python minimal_test.py
```

## 🛠️ Development Tools

- **`backend_ws.py`** - Alternative WebSocket-only backend
- **`main2.py`** - Simplified backend for testing
- **`test.html`** - Frontend test page for training visualization

## 🔒 Security Notes

- JWT secret key should be changed in production
- Database credentials should be properly secured
- CORS is currently set to allow all origins (configure for production)
- Consider implementing rate limiting for production use

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here] 