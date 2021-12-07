import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

n_samples = 100
features, label = make_blobs(n_samples=n_samples, centers=2,\
                             n_features=2,random_state=17)
data = pd.DataFrame({'x':features[:,0],'y':features[:,1],\
                     'z':label})
sns.pairplot(data,hue='z')