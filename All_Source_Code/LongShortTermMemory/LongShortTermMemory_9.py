import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# For LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Load training data
file = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data3.txt'
train = pd.read_csv(file)

# Scale features
s1 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(train[['T1','Q1']])

# Scale predicted value
s2 = MinMaxScaler(feature_range=(-1,1))
Ys = s2.fit_transform(train[['T1']])

# Each time step uses last 'window' to predict the next change
window = 70
X = []
Y = []
for i in range(window,len(Xs)):
    X.append(Xs[i-window:i,:])
    Y.append(Ys[i])

# Reshape data to format accepted by LSTM
X, Y = np.array(X), np.array(Y)

# create and train LSTM model

# Initialize LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, \
          input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error',\
              metrics = ['accuracy'])

# Allow for early exit 
es = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)

# Fit (and time) LSTM model
t0 = time.time()
history = model.fit(X, Y, epochs = 10, batch_size = 250, callbacks=[es], verbose=1)
t1 = time.time()
print('Runtime: %.2f s' %(t1-t0))

# Plot loss 
plt.figure(figsize=(8,4))
plt.semilogy(history.history['loss'])
plt.xlabel('epoch'); plt.ylabel('loss')
plt.savefig('tclab_loss.png')
model.save('model.h5')

# Verify the fit of the model
Yp = model.predict(X)

# un-scale outputs
Yu = s2.inverse_transform(Yp)
Ym = s2.inverse_transform(Y)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(train['Time'][window:],Yu,'r-',label='LSTM')
plt.plot(train['Time'][window:],Ym,'k--',label='Measured')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.subplot(2,1,2)
plt.plot(train['Q1'],label='heater (%)')
plt.legend()
plt.xlabel('Time (sec)'); plt.ylabel('Heater')
plt.savefig('tclab_fit.png')

# Load model
v = load_model('model.h5')
# Load training data
test = pd.read_csv('http://apmonitor.com/pdc/uploads/Main/tclab_data4.txt')

Xt = test[['T1','Q1']].values
Yt = test[['T1']].values

Xts = s1.transform(Xt)
Yts = s2.transform(Yt)

Xti = []
Yti = []
for i in range(window,len(Xts)):
    Xti.append(Xts[i-window:i,:])
    Yti.append(Yts[i])

# Reshape data to format accepted by LSTM
Xti, Yti = np.array(Xti), np.array(Yti)

# Verify the fit of the model
Ytp = model.predict(Xti)

# un-scale outputs
Ytu = s2.inverse_transform(Ytp)
Ytm = s2.inverse_transform(Yti)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(test['Time'][window:],Ytu,'r-',label='LSTM Predicted')
plt.plot(test['Time'][window:],Ytm,'k--',label='Measured')
plt.legend()
plt.ylabel('Temperature (°C)')
plt.subplot(2,1,2)
plt.plot(test['Time'],test['Q1'],'b-',label='Heater')
plt.xlabel('Time (sec)'); plt.ylabel('Heater (%)')
plt.legend()
plt.savefig('tclab_validate.png')

# Using predicted values to predict next step
Xtsq = Xts.copy()
for i in range(window,len(Xtsq)):
    Xin = Xtsq[i-window:i].reshape((1, window, 2))
    Xtsq[i][0] = v.predict(Xin)
    Yti[i-window] = Xtsq[i][0]

#Ytu = (Yti - s2.min_[0])/s2.scale_[0]
Ytu = s2.inverse_transform(Yti)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(test['Time'][window:],Ytu,'r-',label='LSTM Predicted')
plt.plot(test['Time'][window:],Ytm,'k--',label='Measured')
plt.legend()
plt.ylabel('Temperature (°C)')
plt.subplot(2,1,2)
plt.plot(test['Time'],test['Q1'],'b-',label='Heater')
plt.xlabel('Time (sec)'); plt.ylabel('Heater (%)')
plt.legend()
plt.savefig('tclab_forecast.png')
plt.show()