import numpy as np
import matplotlib.pyplot as plt

# Generate a distribution
x = 0.5*np.random.randn(1000)+4

# Standard (mean=0, stdev=1) Scaler
y = (x-np.mean(x))/np.std(x)

# Min-Max (0-1) Scaler
z = (x-np.min(x))/(np.max(x)-np.min(x))

# Plot distributions
plt.figure(figsize=(8,4))
plt.hist(x, bins=30, label='original')
plt.hist(y, alpha=0.7, bins=30, label='standard scaler')
plt.hist(z, alpha=0.7, bins=30, label='minmax scaler')
plt.legend()
plt.show()