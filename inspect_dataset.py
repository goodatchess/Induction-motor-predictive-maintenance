from paderborn_bearing import Paderborn
import numpy as np

data = Paderborn("Healthy", 2048, "Normal")

print("Motor current shape:", np.asarray(data.motor_current).shape)
print("Vibration shape:", np.asarray(data.vibrations).shape)
print("Labels shape:", np.asarray(data.labels).shape)
print("First labels:", np.asarray(data.labels).ravel()[:20])
