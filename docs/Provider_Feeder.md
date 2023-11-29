# Provider_Feeder

A "Databroker Feeder" is a data transmission system that supplies data to a databroker. It collects, transforms, and securely transmits data to the databroker, enabling seamless access and utilization. This feeder uses the "vcan" communication channel for data transmission

```python
import can
from kuksa_client.grpc import VSSClient, Datapoint
import numpy as np
import struct
import time

def receive_can_message(bus):
    message = bus.recv()
    return message

def extract_data_from_message(message):
    speed, steering, throttle, brake, lanedetection = struct.unpack('<eheB?', message.data)
    speed = np.float16(speed)
    steering = np.int16(steering)
    throttle = np.float16(throttle)
    brake = np.uint8(brake)
    lanedetection = bool(lanedetection)
    return speed, steering, throttle, brake, lanedetection

try:
    bus = can.interface.Bus(channel='vcan0', bustype='socketcan')
    with VSSClient('127.0.0.1', 55555) as client:
        while True:
            received_message = receive_can_message(bus)
            speed, steering, throttle, brake, lane = extract_data_from_message(received_message)
            
            client.set_current_values({
                'Vehicle.Speed': Datapoint(speed),
            })
            print(f"Feeding Vehicle.Speed to {speed}")
            
            client.set_current_values({
                'Vehicle.Chassis.SteeringWheel.Angle': Datapoint(steering),
            })
            print(f"Feeding SteeringWheel.Angle to {steering}")
            
            client.set_current_values({
                'Vehicle.OBD.ThrottlePosition': Datapoint(throttle),
            })
            print(f"Feeding ThrottlePosition to {throttle}")
            
            client.set_current_values({
                'Vehicle.Chassis.Brake.PedalPosition': Datapoint(brake),
            })
            print(f"Feeding Brake.PedalPosition to {brake}")
            
            client.set_current_values({
                'Vehicle.ADAS.LaneDepartureDetection.IsWarning': Datapoint(lane),
            })
            print(f"Feeding Lane.Warning to {lane}")

except KeyboardInterrupt:
    print("Interrupted by user, shutting down.")
finally:
    print("Shutting down CAN bus.")
    bus.shutdown()
```
