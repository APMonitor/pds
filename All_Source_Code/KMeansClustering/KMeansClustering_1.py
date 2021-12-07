from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(XA)
yP = km.predict(XB)
# Arbitrary labels with unsupervised clustering may need to be reversed
if len(XB[yP!=yB]) > n/4: yP = 1 - yP 