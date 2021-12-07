from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier()
ab.fit(XA,yA)
yP = ab.predict(XB)