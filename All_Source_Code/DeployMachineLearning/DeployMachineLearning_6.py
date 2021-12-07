# Retrieve model and test data
import pickle
[lr,X_test,y_test] = pickle.load(open('store.pkl','rb'))

# Predict
y_predict = lr.predict(X_test)

# Generate confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cmat = confusion_matrix(y_test,y_predict)
sns.heatmap(cmat,annot=True)
plt.show()