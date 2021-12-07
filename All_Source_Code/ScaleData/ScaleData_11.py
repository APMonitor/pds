from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# split into training (80%) and testing (20%)
train, test = train_test_split(sdata, test_size=0.2, shuffle=True)
train=train.copy(); test=test.copy()

# train neural network
nn = MLPRegressor(hidden_layer_sizes=(3,3),activation='tanh',\
                  solver='lbfgs',max_iter=5000)
model = nn.fit(train[['Q1','T1']],train['T2'])

# test neural network
predict = test.copy()
predict['T2'] = nn.predict(test[['Q1','T1']])

# unscale data
d1 = s.inverse_transform(test)
d2 = s.inverse_transform(predict)
test_results = pd.DataFrame({'T2':d1[:,-1],'T2p':d2[:,-1]})

# plot results
test_results.plot(x='T2',y='T2p',kind='scatter')
plt.plot([15,28],[15,28],'r-')
plt.savefig('results.png',dpi=600)
plt.show()