import numpy as np
import tclab
import time

filename='TCLab_ss1.txt'
fid = open(filename,'w')
fid.write('Q1,Q2,T1,T2\n')
fid.close()

# Connect to Arduino
a = tclab.TCLabModel()

# random heater values
Q1d = np.random.randint(0,70,size=100)
Q2d = np.random.randint(0,80,size=100)

# collect 100 steady state points (~3 minutes each)
print('Wait 180 seconds between heater points')
print('Full data generation requires 5 hrs!')
for i in range(100):
    # set heater values
    a.Q1(Q1d[i])
    a.Q2(Q2d[i])
    # wait 300 seconds
    time.sleep(300)
    print('Set: ' + str(i) + \
          ' Q1: ' + str(Q1d[i]) + \
          ' Q2: ' + str(Q2d[i]) + \
          ' T1: ' + str(a.T1)   + \
          ' T2: ' + str(a.T2))
    fid = open(filename,'a')
    fid.write(str(Q1d[i])+','+str(Q2d[i])+',' \
              +str(a.T1)+','+str(a.T2)+'\n')
    fid.close()
# close connection to Arduino
a.close()