import json
import numpy as np
import matplotlib.pyplot as plt

with open('improve/pac/qrdqn/losses.json') as f:
    Y = json.load(f)
    
X = np.arange(1, 1001)

plt.title("Loss Curve (QR-DQN)")
plt.ylabel('Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.plot(X, Y)
plt.show()