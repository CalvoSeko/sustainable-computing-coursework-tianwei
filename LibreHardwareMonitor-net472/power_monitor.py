import clr # From pip install pythonnet
import time
import os
import sys

# 1. Load the LibreHardwareMonitor Library
# Ensure we are in the correct directory or add it to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # Use full path to ensure it's found
    dll_path = os.path.join(current_dir, "LibreHardwareMonitorLib.dll")
    if not os.path.exists(dll_path):
         raise FileNotFoundError(f"DLL not found at {dll_path}")
    
    clr.AddReference(dll_path)
    from LibreHardwareMonitor.Hardware import Computer
except Exception as e:
    print("Error loading DLL.")
    print(e)
    exit()

# 2. Initialize the Hardware
# Check for Admin privileges
import ctypes
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    print("Warning: Script is not running as Administrator. CPU Power sensors might not be detected.")

computer = Computer()
computer.IsCpuEnabled = True  # Enable CPU sensors
computer.Open()

# 3. Find the Power Sensor (Package Power)
power_sensor = None

# Iterate through hardware to find the CPU
for hardware in computer.Hardware:
    hardware.Update() # Must update to read sensors
    print(f"Checking hardware: {hardware.Name}")
    
    for sensor in hardware.Sensors:
        # Look for "Power" type sensors with "Package" in the name
        if str(sensor.SensorType) == "Power" and "Package" in sensor.Name:
            print(f"  -> FOUND: {sensor.Name} ({sensor.Identifier})")
            power_sensor = sensor
            break
    if power_sensor:
        break

if not power_sensor:
    print("Could not find CPU Package Power sensor!")
    if not is_admin():
        print("-> Try running this script as Administrator.")
    exit()

# 4. Log Data at 100ms Interval
print(f"\nLogging {power_sensor.Name} every 100ms... (Press Ctrl+C to stop)")
start_time = time.time()
next_log_time = start_time + 0.1

try:
    with open("power_log_100ms.csv", "w") as f:
        f.write("Time_Seconds,Watts\n") # Header
        
        while True:
            # Update ONLY the CPU hardware to keep it fast
            power_sensor.Hardware.Update()
            
            # Get timestamp and value
            now = time.time()
            elapsed = now - start_time
            watts = power_sensor.Value
            
            # Write to file (flush immediately so data isn't lost on crash)
            f.write(f"{elapsed:.3f},{watts}\n")
            f.flush() 
            
            # Calculate sleep time to maintain 100ms cadence
            sleep_duration = next_log_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            
            # Set target for next interval
            next_log_time += 0.1

except KeyboardInterrupt:
    print("\nLogging stopped.")
finally:
    computer.Close()