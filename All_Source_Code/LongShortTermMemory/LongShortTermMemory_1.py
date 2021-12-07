import numpy as np
import matplotlib.pyplot as plt

# Generate data
n = 500
t = np.linspace(0,20.0*np.pi,n)
X = np.sin(t) # X is already between -1 and 1, scaling normally needed