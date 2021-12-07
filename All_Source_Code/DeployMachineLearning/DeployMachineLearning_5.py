# Generate blobs dataset
from sklearn.datasets import make_blobs
features, label = make_blobs(n_samples=1000, centers=2,\
                             n_features=2, random_state=12)

# Split into train and test subsets (20% for test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, shuffle=False)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)

# Store model and test data
import pickle
store = [lr,X_test,y_test]
pickle.dump(store,open('store.pkl','wb'))

# View data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.DataFrame({'x1':features[:,0],
                     'x2':features[:,1],
                     'y':label})
sns.pairplot(data,hue='y')
plt.show()