from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler(feature_range=(0,1))
s_train = s.fit_transform(train)
s_test  = s.transform(test)