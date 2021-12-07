from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
sgd.fit(XA,yA)
yP = sgd.predict(XB)