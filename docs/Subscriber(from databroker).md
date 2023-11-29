# Subscriber(from databroker)

This code demonstrates the use of the VSSClient from the kuksa_client.grpc library to subscribe to and handle updates from various vehicle signals in a connected vehicle context. The client connects to a server on the local network at the specified IP address and port. It subscribes to a list of vehicle-related signals such as speed, steering wheel angle, throttle position, brake pedal position, and lane departure warning status. The code continuously listens for updates to these signals, and whenever any signal value changes, it prints out the updated value with a relevant message, indicating real-time monitoring of vehicle status.

```python
from kuksa_client.grpc import VSSClient

with VSSClient('127.0.0.1', 55555) as client:

    # subscribe signals
    signals = [
        'Vehicle.Speed',
        'Vehicle.Chassis.SteeringWheel.Angle',
        'Vehicle.OBD.ThrottlePosition',
        'Vehicle.Chassis.Brake.PedalPosition',
        'Vehicle.ADAS.LaneDepartureDetection.IsWarning'
    ]

    for updates in client.subscribe_current_values(signals):
        # Check the signals updated
        if 'Vehicle.Speed' in updates:
            speed = updates['Vehicle.Speed'].value
            print(f"Received updated speed: {speed}")

        if 'Vehicle.Chassis.SteeringWheel.Angle' in updates:
            steering_angle = updates['Vehicle.Chassis.SteeringWheel.Angle'].value
            print(f"Received updated steering wheel angle: {steering_angle}")

        if 'Vehicle.OBD.ThrottlePosition' in updates:
            throttle_position = updates['Vehicle.OBD.ThrottlePosition'].value
            print(f"Received updated throttle position: {throttle_position}")

        if 'Vehicle.Chassis.Brake.PedalPosition' in updates:
            brake_position = updates['Vehicle.Chassis.Brake.PedalPosition'].value
            print(f"Received updated brake pedal position: {brake_position}")

        if 'Vehicle.ADAS.LaneDepartureDetection.IsWarning' in updates:
            lane_warning = updates['Vehicle.ADAS.LaneDepartureDetection.IsWarning'].value
            print(f"Received lane departure warning: {lane_warning}")
```
