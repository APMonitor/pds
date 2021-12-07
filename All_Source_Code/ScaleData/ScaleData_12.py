from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('http://apmonitor.com/pds/uploads/Main/tclab_data6.txt')
data.set_index('Time',inplace=True)

# split into training (80%) and testing (20%)
train, test = train_test_split(data, test_size=0.2, shuffle=True)
train=train.copy(); test=test.copy()

# train neural network
nn = MLPRegressor(hidden_layer_sizes=(3,3),activation='tanh',\
                  solver='lbfgs',max_iter=5000)
model = nn.fit(train[['Q1','T1']],train['T2'])

# test neural network
test['T2p'] = nn.predict(test[['Q1','T1']])

# plot results
test.plot(x='T2',y='T2p',kind='scatter')
plt.plot([15,28],[15,28],'r-')
plt.show()