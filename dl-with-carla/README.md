# Key concepts

## 1. Make data for model

To obtain driving data, we used Carla, an open-source simulator. We employed two main methods to acquire driving data from Carla:

- Automatic Control
    
    Using the autonomous driving code implemented by Carla, we save five types of data while the vehicle is in motion: speed, steering, throttle, brake, and lane invasion. These data types are stored according to the VSS standard. Depending on the need, this data can be formatted into CSV or input into a virtual CAN.
    
- Manual Control
    
    You can manually drive according to Carla's physics engine, although it is quite challenging. Like in automatic control, the same five data types are saved.
    

## 2. LSTM Learning Model

![lstmmodel.png](/images/lstmmodel.png)

We utilized an LSTM model to calculate driving scores. We chose this model because of its strength in handling time-series data, which is crucial since our CAN data is time-dependent. For more details on the model, please refer to lstm-training.py.

When the five types of data form a sequence of 100, a corresponding score is assigned. With 300,000 CAN data points, a total of 300 data sequences have been trained. Although it's not a vast amount of data, it should be sufficient to represent your driving score!"