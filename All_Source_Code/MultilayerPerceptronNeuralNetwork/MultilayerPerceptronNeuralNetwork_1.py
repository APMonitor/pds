import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model
import matplotlib.pyplot as plt  

# generate training data
x = np.linspace(0.0,2*np.pi,20)
y = np.sin(x)
# save training data to file
data = np.vstack((x,y)).T
np.savetxt('train_data.csv',data,header='x,y',comments='',delimiter=',')

# generate test data
x = np.linspace(0.0,2*np.pi,100)
y = np.sin(x)
# save test data to file
data = np.vstack((x,y)).T
np.savetxt('test_data.csv',data,header='x,y',comments='',delimiter=',')