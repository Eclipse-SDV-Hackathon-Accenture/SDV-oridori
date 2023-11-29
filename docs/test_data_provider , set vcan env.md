# test_data_provider , set vcan env

pre-condition

**CAN-NetModule(https://www.notion.so/test_data_provider-set-vcan-env-7b38b4d799114d01b25b27ee241f78d5?pvs=4#eb945940abe443838413fc8b1c9d6526)**

### **Virtual CAN Interface - vcan(https://netmodule-linux.readthedocs.io/en/latest/howto/can.html#virtual-can-interface-vcan)**

To bring up virtual can interface the kernel module vcan is required. Load vcan module:

```python
modprobe vcan
```

And controls whether the module is loaded successfully:

```python
lsmod | grep vcan
```

Output should be similar to following:

```python
vcan                   16384  0
```

Now a virtual can interface vcan0 can be created:

```python
ip link add dev vcan0 type vcan
ip link set vcan0 mtu 16
ip link set up vcan0
```

To bring up CAN FD interface mtu size must increased to 72:

```python
ip link add dev vcan0 type vcan
ip link set vcan0 mtu 72
ip link set up vcan0
```

And again control new created virtual can interface:

```python
ifconfig vcan0
```

Output should be similar to following:

```python
vcan0     Link encap:UNSPEC  HWaddr 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00
          UP RUNNING NOARP  MTU:16  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
```

 In case the permission denied , use sudo

From this point the virtual can interface vcan0 can be used e.g. for SocketCAN.

```python
import struct
import numpy as np
import math
import can
import os
import time

os.system('sudo ip link set vcan0 down')
os.system('sudo ip link set vcan0 up')
bus = can.interface.Bus(channel='vcan0', bustype='socketcan')

# Example values, replace these with actual values from your data source
c_steer = 0.5  # Example steering coefficient
c_throttle = 0.7  # Example throttle coefficient
c_brake = 0.3  # Example brake coefficient
lanedetection = True  # Example lane detection boolean

while True:
    for v in range(1, 101):  # Loop from 1 to 100
        # Calculate each value
        speed = np.float16(3.6 * math.sqrt(v**2))
        steering = np.int16(c_steer * 90 * v)
        throttle = np.float16(c_throttle * 100)
        brake = np.uint8(c_brake * 100)
        lanedetection = np.bool_(lanedetection)

        # Pack the values into a byte array
        packed_data = struct.pack('<eheB?', speed, steering, throttle, brake, lanedetection)

        message = can.Message(arbitration_id=0x123, data=packed_data, is_extended_id=False)
        bus.send(message)

        # Delay for a while before sending the next message
        time.sleep(1)  # Adjust the delay as needed
```

### .
