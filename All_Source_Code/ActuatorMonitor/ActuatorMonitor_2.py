# generate new data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tclab
import time

n = 3600  # Number of second time points (60 min)
tm = np.linspace(0,n,n+1) # Time values
lab = tclab.TCLab()
T1 = [lab.T1]
T2 = [lab.T2]
Q1 = np.zeros(n+1)
Q2 = np.zeros(n+1)

# random duration on (10-30 sec) in 60 second window
# cool down last 5 minutes
k = 60
for i in range(1,n-301):
    if (i%k)==0:
        j = np.random.randint(10,26)
        k = np.random.randint(5,180)
        Q1[i:i+j+1] = 100.0
for i in range(n):
    lab.Q1(Q1[i])
    lab.Q2(Q2[i])
    time.sleep(1)
    print(Q1[i],lab.T1)
    T1.append(lab.T1)
    T2.append(lab.T2)
lab.close()
# Save data file
data = np.vstack((tm,Q1,Q2,T1,T2)).T
np.savetxt('tclab_train.csv',data,delimiter=',',\
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
plt.savefig('tclab_train.png')
plt.show()