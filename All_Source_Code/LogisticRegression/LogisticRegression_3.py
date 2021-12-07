import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns

# Generate blobs dataset
features, label = make_blobs(n_samples=800, centers=2,\
                             n_features=2, random_state=12) 
data = pd.DataFrame()
data['x1'] = features[:,0]
data['x2'] = features[:,1]
data['y']  = label

sns.pairplot(data,hue='y')