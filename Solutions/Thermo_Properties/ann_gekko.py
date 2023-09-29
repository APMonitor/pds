from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('group_contrib_data.csv')

print(data.head())

# Size with hyperbolic tangent function
nin = 40  # inputs
n1 = 1   # hidden layer 1 (linear)
n2 = 5   # hidden layer 2 (nonlinear)
n3 = 1   # hidden layer 3 (linear)
nout = 1 # outputs

# input
d = np.array(data)[:,-40:]
d = np.array(d,dtype=float)

s_in_min = np.min(d,axis=0)
s_in_range = np.max(d,axis=0)-np.min(d,axis=0)
ds = np.empty_like(d)
for i in range(nin):
    for j in range(np.size(d,0)):
        ds[j,i] = (d[j,i]-s_in_min[i])/s_in_range[i]
ds_train = ds[0:205]
ds_valid = ds[205:]

# measured output
meas = np.array(data['Par (calc./est.)'])
s_out_min = np.min(meas)
s_out_range = np.max(meas)-np.min(meas)
meass = np.empty_like(meas)
for j in range(len(meas)):
    meass[j] = (meas[j] - s_out_min) / s_out_range 
# scaled values
meas_train = meass[0:205]
meas_valid = meass[205:]
# unscaled values
umeas_train = meas[0:205]
umeas_valid = meas[205:]

# Initialize gekko
train = GEKKO(remote=False) 
test = GEKKO(remote=False)

model = [train,test]

for m in model:
    # input(s)
    m.inpt = [m.Param() for i in range(nin)]

    # layer 1
    m.w1 = m.Array(m.FV, (nin,n1))
    m.l1 = [sum([m.Intermediate(m.w1[j,i]*m.inpt[j]) for j in range(nin)]) for i in range(n1)]

    # layer 2
    m.w2 = m.Array(m.FV, (n1,n2))
    m.l2 = [m.Intermediate(sum([m.tanh(m.w2[j,i]*m.l1[j]) \
                           for j in range(n1)])) for i in range(n2)]

    # layer 3
    m.w3 = m.Array(m.FV, (n2,n3))
    m.l3 = [m.Intermediate(sum([m.w3[j,i]*m.l2[j] \
            for j in range(n2)])) for i in range(n3)]

    # output(s)
    m.outpt = m.CV()
    m.Equation(m.outpt==sum([m.l3[i] for i in range(n3)]))

    # flatten matrices
    m.w1 = m.w1.flatten()
    m.w2 = m.w2.flatten()
    m.w3 = m.w3.flatten()

# Fit parameter weights
m = train
for i in range(nin):
    m.inpt[i].value=ds_train[:,i]
m.outpt.value=meas_train
m.outpt.FSTATUS = 1
for i in range(len(m.w1)):
    m.w1[i].FSTATUS=1
    m.w1[i].STATUS=1
    m.w1[i].MEAS=1.0
    m.w1[i].LOWER = 0.0 # add constraint
for i in range(len(m.w2)):
    m.w2[i].STATUS=1
    m.w2[i].FSTATUS=1
    m.w2[i].MEAS=0.5
for i in range(len(m.w3)):
    m.w3[i].FSTATUS=1
    m.w3[i].STATUS=1
    m.w3[i].MEAS=1.0
m.options.IMODE = 2
#m.options.OTOL = 1e-2
m.options.SOLVER = 1
m.options.EV_TYPE = 2
m.options.MAX_ITER = 1000
m.solve(disp=True)

# Test sample points
m = test
for i in range(len(m.w1)):
    m.w1[i].MEAS=train.w1[i].NEWVAL
    m.w1[i].FSTATUS = 1
    print('w1['+str(i)+']: '+str(m.w1[i].MEAS))
for i in range(len(m.w2)):
    m.w2[i].MEAS=train.w2[i].NEWVAL
    m.w2[i].FSTATUS = 1
    print('w2['+str(i)+']: '+str(m.w2[i].MEAS))
for i in range(len(m.w3)):
    m.w3[i].MEAS=train.w3[i].NEWVAL
    m.w3[i].FSTATUS = 1
    print('w3['+str(i)+']: '+str(m.w3[i].MEAS))
for i in range(nin):
    m.inpt[i].value=ds_valid[:,i]
m.options.IMODE = 2
m.options.SOLVER = 1
m.solve(disp=True)

# unscale
pred_train = np.empty_like(meas_train)
for i in range(len(pred_train)):
    pred_train[i] = train.outpt.value[i] * s_out_range + s_out_min

pred_valid = np.empty_like(meas_valid)
for i in range(len(pred_valid)):
    pred_valid[i] = test.outpt.value[i] * s_out_range + s_out_min

print('ms_abs train')
print(np.sum(np.abs((umeas_train-pred_train)/umeas_train)/(len(umeas_train))))
print('ms_abs validate')
print(np.sum(np.abs((umeas_valid-pred_valid)/umeas_valid)/(len(umeas_valid))))

# parity plot
plt.loglog([80,2000],[80,2000],'k-')
plt.loglog(umeas_train,pred_train,'b.',label='ANN (Train)')
plt.loglog(umeas_valid,pred_valid,'r.',label='ANN (Validate)')
plt.legend()
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.savefig('ann_gekko.png')
plt.show()
