# Balanced Data
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

feature, label = make_blobs(n_samples=[2000,2000],\
                       n_features=2,\
                       centers=[(-5,5),(5,5)],\
                       random_state=47,\
                       cluster_std=3)
plt.figure(figsize=(6,3))
for cv in range(2):
    row = np.where(label==cv)
    plt.scatter(feature[row,0],feature[row,1],\
                cmap='Paired')
plt.tight_layout()
plt.savefig('balanced_data.png',dpi=300)
plt.show()