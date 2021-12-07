import tclab
import time
import numpy as np
from simple_pid import PID
import matplotlib.pyplot as plt

# Create PID controller
pid = PID(Kp=5.0,Ki=0.05,Kd=1.0,\
          setpoint=50,sample_time=1.0,\
          output_limits=(0,100))

n = 300
tm = np.linspace(0,n-1,n)
T1 = np.zeros(n); Q1 = np.zeros(n)

lab = tclab.TCLab()
for i in range(n):
    # read temperature
    T1[i] = lab.T1

    # PID control
    Q1[i] = pid(T1[i]) 
    lab.Q1(Q1[i])

    # print
    if i%50==0:
        print('Time OP PV   SP')
    if i%5==0:
        print(i,round(Q1[i],2), T1[i], pid.setpoint)
    # wait sample time
    time.sleep(pid.sample_time) # wait 1 sec
lab.close()