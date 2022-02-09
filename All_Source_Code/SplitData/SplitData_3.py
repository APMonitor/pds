from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# define dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15)

# Set up K-fold cross validation
kf = KFold(n_splits=5,shuffle=True)

# Initialize model
dtc = DecisionTreeClassifier()

# Array to store accuracy scores
scores = np.zeros(5)

# Initialize plot
plt.figure(figsize=(12,2))

for i,(train_index, test_index) in enumerate(kf.split(X)):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    dtc.fit(Xtrain,ytrain)
    yp = dtc.predict(Xtest)
    acc = accuracy_score(ytest,yp)
    scores[i] = acc

    plt.subplot(1,5,i+1)
    cm = confusion_matrix(yp,ytest)
    sns.heatmap(cm,annot=True)

plt.show()
print('Accuracy: %.2f%%' %(np.mean(scores*100)))