from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and parse into columns
url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)
t = data['Time']
u = data[['H1','H2']]
y = data[['T1','T2']]

m = GEKKO()

# system identification
na = 2 # output coefficients
nb = 2 # input coefficients
yp,p,K = m.sysid(t,u,y,na,nb,pred='meas')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,u,label=r'$Heater_1$')
plt.legend([r'$Heater_1$',r'$Heater_2$'])
plt.ylabel('Heaters')
plt.subplot(2,1,2)
plt.plot(t,y)
plt.plot(t,yp,'--')
plt.legend([r'$T1_{meas}$',r'$T2_{meas}$',\
            r'$T1_{pred}$',r'$T2_{pred}$'])
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (sec)')
plt.show()