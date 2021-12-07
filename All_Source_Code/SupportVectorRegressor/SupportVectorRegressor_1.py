from sklearn import svm
s = svm.SVR(gamma='scale')
s.fit(XA,yA)
yP = s.predict(XB)