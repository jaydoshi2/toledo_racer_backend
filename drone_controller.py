import asyncio
import threading
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DroneCommandListener(Node):
    def __init__(self):
        super().__init__('drone_command_listener')
        self.subscription = self.create_subscription(
            String,
            '/control_commands',
            self.listener_callback,
            10
        )
        print("[INFO] Drone command listener initialized. Waiting for commands...")
        
        self.drone = System()
        self.drone_connected = False
        self.offboard_active = False
        self.drone_loop = None
        
        # Start the async drone connection in a dedicated thread with its own event loop
        self.drone_thread = threading.Thread(target=self.start_drone_thread)
        self.drone_thread.daemon = True
        self.drone_thread.start()

    def start_drone_thread(self):
        """Create a dedicated event loop for drone operations"""
        self.drone_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.drone_loop)
        try:
            self.drone_loop.run_until_complete(self.connect_drone())
        except Exception as e:
            print(f"[ERROR] Drone thread error: {e}")

    async def connect_drone(self):
        """Connect to PX4 SITL"""
        try:
            print("[INFO] Connecting to PX4 SITL on udp://:14540...")
            await self.drone.connect(system_address="udp://:14540")
            
            print("[INFO] Waiting for drone connection...")
            async for state in self.drone.core.connection_state():
                print(f"[DEBUG] Connection state: {state.is_connected}")
                if state.is_connected:
                    print("[✅] Drone connection established!")
                    self.drone_connected = True
                    break
                    
            if self.drone_connected:
                print("[INFO] Drone is ready for commands!")
                # Keep the event loop running to handle commands
                await self.keep_loop_alive()
                    
        except Exception as e:
            print(f"[ERROR] Failed to connect to drone: {e}")

    async def keep_loop_alive(self):
        """Keep the drone event loop alive to handle incoming commands"""
        while True:
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    def listener_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"Received command: {command}")
        
        if self.drone_connected and self.drone_loop:
            # Schedule the command in the drone's event loop
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.handle_command(command), 
                    self.drone_loop
                )
                # Optional: You can wait for completion with timeout
                # future.result(timeout=5.0)
            except Exception as e:
                print(f"[ERROR] Failed to schedule command: {e}")
        else:
            self.get_logger().warn("Drone not connected yet.")

    async def handle_command(self, command):
        try:
            if command == "takeoff":
                print("[INFO] Arming and taking off...")
                await self.drone.action.arm()
                print("[INFO] Armed successfully!")
                await self.drone.action.takeoff()
                print("[✅] Takeoff command completed!")
                
            elif command == "land":
                print("[INFO] Landing...")
                if self.offboard_active:
                    await self.drone.offboard.stop()
                    self.offboard_active = False
                    print("[INFO] Offboard mode stopped")
                await self.drone.action.land()
                print("[✅] Land command completed!")
                
            elif command in ["move_forward", "move_backward", "turn_left", "turn_right", "stop"]:
                if not self.offboard_active:
                    print("[INFO] Starting offboard mode...")
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
                    await self.drone.offboard.start()
                    self.offboard_active = True
                    print("[INFO] Offboard mode started!")
                
                if command == "move_forward":
                    print("[INFO] Moving forward...")
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(2.0, 0.0, 0.0, 0.0))
                    
                elif command == "move_backward":
                    print("[INFO] Moving backward...")
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(-2.0, 0.0, 0.0, 0.0))
                    
                elif command == "turn_left":
                    print("[INFO] Turning left...")
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, -30.0))
                    
                elif command == "turn_right":
                    print("[INFO] Turning right...")
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 30.0))
                    
                elif command == "stop":
                    print("[INFO] Stopping...")
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
                    
                print(f"[✅] Command '{command}' completed!")
                    
            else:
                self.get_logger().warn(f"Unknown command: {command}")
                
        except Exception as e:
            print(f"[ERROR] Failed to execute command '{command}': {e}")

def main():
    rclpy.init()
    
    try:
        node = DroneCommandListener()
        print("[INFO] Drone controller started. Use Ctrl+C to exit.")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
        
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()