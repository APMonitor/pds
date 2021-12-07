import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8)
Xa,Xb,ya,yb = train_test_split(X, y, test_size=0.2, shuffle=True)
xgbc = xgb.XGBClassifier()
xgbc.fit(Xa,ya)
yp = xgbc.predict(Xb)
acc = accuracy_score(yb,yp)
print(acc)
cm = confusion_matrix(yp,yb)
sns.heatmap(cm,annot=True)
plt.show()