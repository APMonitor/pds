import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import pandas as pd

file = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data3.txt'
data = pd.read_csv(file)

# subset for training
n = 3000
tm = data['Time'][0:n].values
Q1s = data['Q1'][0:n].values
T1s = data['T1'][0:n].values

m = GEKKO()
m.time = tm

# Parameters to Estimate
K1 = m.FV(value=0.5,lb=0.1,ub=1.0)
tau1 = m.FV(value=150,lb=50,ub=250)
tau2 = m.FV(value=15,lb=10,ub=20)
K1.STATUS = 1
tau1.STATUS = 1
tau2.STATUS = 1

# Model Inputs
Q1 = m.Param(value=Q1s)
Ta = m.Param(value=23.0) # degC
T1m = m.Param(T1s)

# Model Variables
TH1 = m.Var(value=T1s[0])
TC1 = m.Var(value=T1s)

# Objective Function
m.Minimize((T1m-TC1)**2)

# Equations
m.Equation(tau1 * TH1.dt() + (TH1-Ta) == K1*Q1)
m.Equation(tau2 * TC1.dt()  + TC1 == TH1)

# Global Options
m.options.IMODE   = 5 # MHE
m.options.EV_TYPE = 2 # Objective type
m.options.NODES   = 2 # Collocation nodes
m.options.SOLVER  = 3 # IPOPT

# Predict Parameters and Temperatures
m.solve() 

# Create plot
plt.figure(figsize=(10,7))

ax=plt.subplot(2,1,1)
ax.grid()
plt.plot(tm,T1s,'ro',label=r'$T_1$ measured')
plt.plot(tm,TC1.value,'k-',label=r'$T_1$ predicted')
plt.ylabel('Temperature (degC)')
plt.legend(loc=2)
ax=plt.subplot(2,1,2)
ax.grid()
plt.plot(tm,Q1s,'b-',label=r'$Q_1$')
plt.ylabel('Heater (%)')
plt.xlabel('Time (sec)')
plt.legend(loc='best')

# Print optimal values
print('K1: ' + str(K1.newval))
print('tau1: ' + str(tau1.newval))
print('tau2: ' + str(tau2.newval))

# Save and show figure
plt.savefig('tclab_2nd_order_fit.png')


# Validation
tm = data['Time'][n:3*n].values
Q1s = data['Q1'][n:3*n].values
T1s = data['T1'][n:3*n].values

v = GEKKO()
v.time = tm

# Parameters to Estimate
K1 = K1.newval
tau1 = tau1.newval
tau2 = tau2.newval
Q1 = v.Param(value=Q1s)
Ta = v.Param(value=23.0) # degC
TH1 = v.Var(value=T1s[0])
TC1 = v.Var(value=T1s[0])
v.Equation(tau1 * TH1.dt() + (TH1-Ta) == K1*Q1)
v.Equation(tau2 * TC1.dt()  + TC1 == TH1)
v.options.IMODE   = 4 # Simulate
v.options.NODES   = 2 # Collocation nodes
v.options.SOLVER  = 1

# Predict Parameters and Temperatures
v.solve(disp=True) 

# Create plot
plt.figure(figsize=(10,7))

ax=plt.subplot(2,1,1)
ax.grid()
plt.plot(tm,T1s,'ro',label=r'$T_1$ measured')
plt.plot(tm,TC1.value,'k-',label=r'$T_1$ predicted')
plt.ylabel('Temperature (degC)')
plt.legend(loc=2)
ax=plt.subplot(2,1,2)
ax.grid()
plt.plot(tm,Q1s,'b-',label=r'$Q_1$')
plt.ylabel('Heater (%)')
plt.xlabel('Time (sec)')
plt.legend(loc='best')

# Save and show figure
plt.savefig('tclab_2nd_order_validate.png')
plt.show()