import asyncio
from mavsdk import System

async def connect_and_arm():
    """Connects to PX4 SITL and attempts to arm the vehicle."""
    drone = System()
    await drone.connect(system_address="udp://:14540") # Connect to SITL on default UDP port

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("Arming...")
    try:
        await drone.action.arm()
        print("Armed!")
    except Exception as e:
        print(f"Failed to arm: {e}")

    # Example: Takeoff (optional)
    # print("Taking off...")
    # await drone.action.takeoff()

    # You can now send other commands, e.g., move to a specific location, land, etc.
    # await drone.action.land()

if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(connect_and_arm())