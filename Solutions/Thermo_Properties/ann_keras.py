import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt  

#################################################################
### Import Data #################################################
#################################################################

data = pd.read_csv('group_contrib_data.csv')

# input
d = np.array(data)[:,-40:]
d = np.array(d,dtype=float)
x_train = d[0:205]
x_valid = d[205:]

# measured output
meas = np.array(data['Par (calc./est.)'])
y_train = meas[0:205]
y_valid = meas[205:]

train = np.vstack((x_train.T,y_train)).T
valid = np.vstack((x_valid.T,y_valid)).T

#################################################################
### Scale data ##################################################
#################################################################

# scale values to 0 to 1 for the ANN to work well
s = MinMaxScaler(feature_range=(0,1))

# scale training and test data
sc_train = s.fit_transform(train)
xs_train = sc_train[:,0:-1]
ys_train = sc_train[:,-1]

sc_valid = s.transform(valid)
xs_valid = sc_valid[:,0:-1]
ys_valid = sc_valid[:,-1]

#################################################################
### Train model #################################################
#################################################################

# create neural network model
model = Sequential()
model.add(Dense(40, input_dim=40, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")

# load training data
X1 = xs_train
Y1 = ys_train

# train the model
model.fit(X1,Y1,epochs=200,verbose=1,shuffle=True)

#################################################################
### Test model ##################################################
#################################################################

# load test data
X2 = xs_valid
Y2 = ys_valid

# test the model
mse_train = model.evaluate(X1,Y1, verbose=1)
mse_valid = model.evaluate(X2,Y2, verbose=1)

print('Mean Squared Error (Train): ', mse_train)
print('Mean Squared Error (Valid): ', mse_valid)

#################################################################
### Predictions Outside Training Region #########################
#################################################################

# predict
Y1P = model.predict(X1)
Y2P = model.predict(X2)

# unscale for plotting and analysis
ymin = s.min_[-1]
yrange = s.scale_[-1]

Y1u = (Y1-ymin)/yrange
Y1Pu = (Y1P-ymin)/yrange

Y2u = (Y2-ymin)/yrange
Y2Pu = (Y2P-ymin)/yrange

sae1 = 0.0
for i in range(len(Y1u)):
    sae1 += np.abs(Y1u[i]-Y1Pu[i][0])/Y1u[i]
sae1 = sae1 / len(Y1u)

sae2 = 0.0
for i in range(len(Y2u)):
    sae2 += np.abs(Y2u[i]-Y2Pu[i][0])/Y2u[i]
sae2 = sae2 / len(Y2u)

# mean sum abs difference
print('Mean sum abs diff - Training ' + str(sae1))
print('Mean sum abs diff - Validate ' + str(sae2))

plt.figure()
plt.plot(Y1u, Y1Pu, 'b.',label='train')
plt.plot(Y2u, Y2Pu, 'r.',label='validate')
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.legend(loc='best')
plt.savefig('results_keras.png')
plt.show()
