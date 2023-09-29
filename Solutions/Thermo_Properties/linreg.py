import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('group_contrib_data.csv')

print(data.head())

# input
d = np.array(data)[:,-40:]
d = np.array(d,dtype=float)

d_train = d[0:205]
d_valid = d[205:]

# measured output
meas = np.array(data['Par (calc./est.)'])
meas_train = meas[0:205]
meas_valid = meas[205:]

# linear regression
#  d * b = p
#  (d^T * d) * b = (d^T * meas)
#  A * b = rhs
A = np.dot(d_train.T,d_train)
rhs = np.dot(d_train.T,meas_train)
# solve for
#  b = inv(d^T*d)*d^T*p
b = np.linalg.solve(A,rhs)

# predicted output
pred_train = np.dot(d_train,b)
pred_valid = np.dot(d_valid,b)

print('ms_abs train')
print(np.sum(np.abs((meas_train-pred_train)/meas_train)/(len(meas_train))))
print('ms_abs validate')
print(np.sum(np.abs((meas_valid-pred_valid)/meas_valid)/(len(meas_valid))))

# parity plot
plt.loglog([80,2000],[80,2000],'k-')
plt.loglog(meas_train,pred_train,'b.',label='Linear (Train)')
plt.loglog(meas_valid,pred_valid,'r.',label='Linear (Validate)')
plt.legend()
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.savefig('linreg.png')
plt.show()
