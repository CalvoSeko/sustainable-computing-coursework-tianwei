import clr
import time
import os
import sys
import threading
from functools import wraps

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

dll_path = os.path.join(current_dir, "LibreHardwareMonitorLib.dll")
clr.AddReference(dll_path)
from LibreHardwareMonitor.Hardware import Computer


def get_power_sensors():
    """Return (computer, list_of_sensors) for all sensors with SensorType 'Power'."""
    computer = Computer()
    # enable common hardware; guard with hasattr
    for attr in ("IsCpuEnabled", "IsGpuEnabled", "IsMemoryEnabled", "IsMotherboardEnabled", "IsStorageEnabled"):
        if hasattr(computer, attr):
            setattr(computer, attr, True)
    computer.Open()

    sensors = []
    for hardware in computer.Hardware:
        hardware.Update()
        for sensor in hardware.Sensors:
            if str(sensor.SensorType) == "Power":
                sensors.append(sensor)

    if not sensors:
        computer.Close()
        raise RuntimeError("No power sensors found. Try running as Administrator.")

    return computer, sensors


class PowerMonitor:
    """Measure power for all discovered power sensors.

    CSV format: Time_Seconds,Sensor_Name,Sensor_ID,Watts
    """

    def __init__(self, filename="power_log.csv", interval=0.1):
        self.filename = filename
        self.interval = interval
        self.computer = None
        self.sensors = []
        self.data = []  # list of (elapsed, [(name,id,value), ...])
        self.running = False

    def start(self):
        self.computer, self.sensors = get_power_sensors()
        self.data = []
        self.running = True
        self.start_time = time.time()
        threading.Thread(target=self._monitor, daemon=True).start()

    def _monitor(self):
        next_time = self.start_time + self.interval
        while self.running:
            elapsed = time.time() - self.start_time
            row = []
            for s in self.sensors:
                # update the hardware the sensor belongs to
                try:
                    s.Hardware.Update()
                except Exception:
                    pass
                row.append((s.Name, str(s.Identifier), s.Value))
            self.data.append((elapsed, row))

            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_time += self.interval

    def stop(self):
        self.running = False
        if self.computer:
            self.computer.Close()

        # write per-sensor rows
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("Time_Seconds,Sensor_Name,Sensor_ID,Watts\n")
            for elapsed, rows in self.data:
                for name, sid, watts in rows:
                    f.write(f"{elapsed:.3f},{name},{sid},{watts}\n")

    def stats(self):
        if not self.data:
            return None
        per_sensor = {}
        for _, rows in self.data:
            for name, sid, watts in rows:
                per_sensor.setdefault(name, []).append(watts)

        out = {}
        for name, vals in per_sensor.items():
            out[name] = {
                "min": min(vals),
                "max": max(vals),
                "avg": sum(vals) / len(vals),
                "samples": len(vals)
            }
        return out

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out_name = f"power_{func.__name__}.csv"
            with PowerMonitor(filename=out_name, interval=self.interval):
                return func(*args, **kwargs)
        return wrapper


# Quick example when run directly
if __name__ == "__main__":
    with PowerMonitor("power_all.csv") as m:
        time.sleep(2)
    print(m.stats())