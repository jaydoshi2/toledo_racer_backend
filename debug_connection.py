import asyncio
import socket
from mavsdk import System

async def test_connection():
    # First, test if the port is reachable
    print("Testing UDP port 14540...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 14540))
        sock.close()
        print(f"Port 14540 connection test result: {result}")
    except Exception as e:
        print(f"Port test failed: {e}")

    # Test different connection strings
    connection_strings = [
        "udp://:14540",
        "udp://127.0.0.1:14540",
        "udp://localhost:14540",
        "tcp://127.0.0.1:4560",
        "serial:///dev/ttyACM0:57600"  # Sometimes SITL uses serial
    ]
    
    for conn_str in connection_strings:
        print(f"\n--- Testing connection: {conn_str} ---")
        drone = System()
        
        try:
            await drone.connect(system_address=conn_str)
            print(f"Connection command sent to {conn_str}")
            
            # Wait with timeout
            import time
            start_time = time.time()
            timeout = 5  # 5 seconds timeout
            
            async for state in drone.core.connection_state():
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Connection state: {state.is_connected}")
                
                if state.is_connected:
                    print(f"✅ SUCCESS: Connected via {conn_str}")
                    
                    # Test basic telemetry
                    try:
                        async for position in drone.telemetry.position():
                            print(f"Position: lat={position.latitude_deg}, lon={position.longitude_deg}")
                            return conn_str  # Return successful connection string
                    except Exception as e:
                        print(f"Telemetry error: {e}")
                        return conn_str
                
                if elapsed > timeout:
                    print(f"❌ TIMEOUT: {conn_str} after {timeout}s")
                    break
                    
        except Exception as e:
            print(f"❌ ERROR: {conn_str} failed: {e}")
            continue
    
    print("❌ All connection attempts failed")
    return None

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    if result:
        print(f"\n✅ Use this connection string: {result}")
    else:
        print("\n❌ No working connection found")