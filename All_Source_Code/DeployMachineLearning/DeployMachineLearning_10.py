from os.path import exists
import pickle
import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

if exists('m.pkl'):
    # load model from subsequent call
    m = pickle.load(open('m.pkl','rb'))
    m.solve()
else:
    # define model the first time
    m = GEKKO()
    m.time = np.linspace(0,20,41)

    m.p = m.MV(value=0, lb=0, ub=1)
    m.v = m.CV(value=0)
    m.Equation(5*m.v.dt() == -m.v + 10*m.p)
    m.options.IMODE = 6
    m.p.STATUS = 1; m.p.DCOST = 1e-3
    m.v.STATUS = 1; m.v.SP = 40; m.v.TAU = 5
    m.options.CV_TYPE = 2
    m.solve()
pickle.dump(m,open('m.pkl','wb'))

plt.figure()
plt.subplot(2,1,1)
plt.plot(m.time,m.p.value,'b-',lw=2)
plt.ylabel('gas')
plt.subplot(2,1,2)
plt.plot(m.time,m.v.value,'r--',lw=2)
plt.ylabel('velocity')
plt.xlabel('time')
plt.show()