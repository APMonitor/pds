import tclab
import time
import numpy as np
from simple_pid import PID
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model

h5_file,s_x,s_y,window = pickle.load(open('lstm_control.pkl','rb'))
model = load_model(h5_file)

def lstm(T1_m, Tsp_m):
    # Calculate error (necessary feature for LSTM input)
    err = Tsp_m - T1_m
    
    # Format data for LSTM input
    X = np.vstack((Tsp_m,err)).T
    Xs = s_x.transform(X)
    Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))
    
    # Predict Q for controller and unscale
    Q1c_s = model.predict(Xs)
    Q1c = s_y.inverse_transform(Q1c_s)[0][0]
    
    # Ensure Q1c is between 0 and 100
    Q1c = np.clip(Q1c,0.0,100.0)
    return Q1c

n = 300
tm = np.linspace(0,n-1,n)
T1 = np.zeros(n); Q1 = np.zeros(n)

lab = tclab.TCLabModel()
for i in range(n):
    # read temperature
    T1[i] = lab.T1

    # LSTM control
    if i>=window:
        T1_m = T1[i-window:i]
    else:
        insert = np.ones(window-i)*T1[0]
        T1_m = np.concatenate((insert,T1[0:i]))
    Tsp_m = 50*np.ones(window)
    Q1[i] = lstm(T1_m,Tsp_m) 

    # implement on the TCLab
    lab.Q1(Q1[i])

    if i%50==0:
        print('Time     OP     PV   SP')
    if i%5==0:
        print("{0:4d} {1:6.2f} {2:6.2f} {3:4d}"\
              .format(i,Q1[i],T1[i],50))
    # wait sample time
    time.sleep(1) # wait 1 sec
lab.close()

# Create Figure
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.plot([0,tm[-1]/60.0],[50,50],'k-',label=r'$T_1$ SP')
plt.plot(tm/60.0,T1,'r.',label=r'$T_1$ PV')
plt.ylabel(r'Temp ($^oC$)')
plt.legend()
plt.subplot(2,1,2)
plt.grid()
plt.plot(tm/60.0,Q1,'b-',label=r'$Q_1$')
plt.ylabel(r'Heater (%)'); plt.xlabel('Time (min)')
plt.legend()
plt.show()
