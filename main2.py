# backend.py
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

rclpy.init()

class DronePublisher(Node):
    def __init__(self):
        super().__init__('drone_command_publisher')
        self.publisher_ = self.create_publisher(String, '/control_commands', 10)

    def publish_command(self, command: str):
        msg = String()
        msg.data = command
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')

publisher_node = DronePublisher()

def ros_spin():
    rclpy.spin(publisher_node)

threading.Thread(target=ros_spin, daemon=True).start()

class CommandRequest(BaseModel):
    command: str

@app.post("/send-command/")
def send_command(request: CommandRequest):
    publisher_node.publish_command(request.command)
    return {"status": "sent", "command": request.command}


# Update model status and metrics
# @app.put("/models/{model_id}", response_model=DroneModel)
# def update_model(model_id: int, update_data: DroneModelUpdate, db: Session = Depends(get_db)):
#     db_model = update_drone_model(db, model_id=model_id, update_data=update_data)
#     if db_model is None:
#         raise HTTPException(status_code=404, detail="Model not found")
#     return db_model

# # Get all models for a usergive me the 
# @app.get("/users/{username}/models/", response_model=List[DroneModel])
# def get_models(username: str, db: Session = Depends(get_db)):
#     db_user = get_user_by_username(db, username=username)
#     if db_user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return get_user_models(db=db, user_id=db_user.id)

# # Get model by drone_id
# @app.get("/models/drone/{drone_id}", response_model=DroneModel)
# def get_model_by_drone(drone_id: str, db: Session = Depends(get_db)):
#     print(f"Getting model by drone_id {drone_id}")
#     db_model = get_model_by_drone_id(db, drone_id=drone_id)
#     if db_model is None:
#         raise HTTPException(status_code=404, detail="Model not found")
#     return db_model

# # WebSocket endpoint for training updates
# @app.websocket("/ws/train/{model_id}")
# async def train_model(websocket: WebSocket, mraceback (most recent call last):
#     print(f"Training model {model_id}")
#     await websocket.accept()
    
#     try:
#         # Get model from database
#         db_model = get_model_by_id(db, model_id)
#         if not db_model:
#             await websocket.send_json({"error": "Model not found"})
#             await websocket.close()
#             return

#         # Update status to training
#         update_data = DroneModelUpdate(status="training")
#         update_drone_model(db, model_id, update_data)

#         # Simulate training
#         for epoch in range(db_model.training_epochs):
#             # Simulate training metrics
#             loss = 1.0 - (epoch / db_model.training_epochs)  # Simulated decreasing loss
#             accuracy = epoch / db_model.training_epochs  # Simulated increasing accuracy
            
#             metrics = {
#                 "epoch": epoch + 1,
#                 "total_epochs": db_model.training_epochs,
#                 "loss": round(loss, 4),
#                 "accuracy": round(accuracy, 4),
#                 "progress": round((epoch + 1) / db_model.training_epochs * 100, 1),
#                 "done": False
#             }
            
#             await websocket.send_json(metrics)
#             await asyncio.sleep(0.5)  # Simulate training time

#         # Update final status and metrics
#         final_update = DroneModelUpdate(
#             status="finished",
#             train_loss=0.1,  # Final simulated loss
#             train_accuracy=0.95  # Final simulated accuracy
#         )
#         update_drone_model(db, model_id, final_update)
        
#         await websocket.send_json({
#             "done": True,
#             "message": "Training completed!",give me the 
#             "final_loss": 0.1,
#             "final_accuracy": 0.95
#         })
        
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print(e)
#         await websocket.send_json({"error": str(e), "done": True})
#     finally:
#         await websocket.close()
