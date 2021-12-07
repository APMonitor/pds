# generate new data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tclab
import time

n = 600  # Number of second time points (10 min)
tm = np.linspace(0,n,n+1) # Time values
lab = tclab.TCLab()
T1 = [lab.T1]
T2 = [lab.T2]
Q1 = np.zeros(n+1)
Q2 = np.zeros(n+1)
# turn on heater at select times
Q1[20:41]   = 100.0; Q1[60:91]   = 100.0; Q1[150:181] = 100.0
Q1[190:206] = 100.0; Q1[220:251] = 100.0; Q1[260:291] = 100.0
Q1[300:316] = 100.0; Q1[340:351] = 100.0; Q1[400:431] = 100.0
Q1[500:521] = 100.0; Q1[540:571] = 100.0
for i in range(n):
    lab.Q1(Q1[i]); lab.Q2(Q2[i])
    time.sleep(1)
    print(Q1[i],lab.T1)
    T1.append(lab.T1); T2.append(lab.T2)
lab.close()
# Save data file
data = np.vstack((tm,Q1,Q2,T1,T2)).T
np.savetxt('tclab_test.csv',data,delimiter=',',\
           header='Time,Q1,Q2,T1,T2',comments='')

# Create Figure
plt.figure(figsize=(10,7))
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(tm/60.0,T1,'r.',label=r'$T_1$')
plt.ylabel(r'Temp ($^oC$)')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(tm/60.0,Q1,'b-',label=r'$Q_1$')
plt.ylabel(r'Heater (%)')
plt.xlabel('Time (min)')
plt.legend()
plt.savefig('tclab_test.png')
plt.show()