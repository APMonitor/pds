yhat = []
for row in test:
    yhat.append(round(predict(row, beta)))

from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(test[:,-1],yhat)
sns.heatmap(cmat,annot=True)