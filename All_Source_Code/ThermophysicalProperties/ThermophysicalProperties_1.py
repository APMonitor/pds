from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# define data
x = np.random.rand(100) + np.linspace(0,100,100)
y = np.random.rand(100) - 2*np.linspace(0,1,100)

# linear regression model with Gekko
m = GEKKO()

# unknown parameters
a,b = m.Array(m.FV,2)
a.STATUS = 1; b.STATUS = 1

# variables and parameters
yp = m.Var()
ym = m.Param(y)
xm = m.Param(x)

# equations and objective
m.Equation(yp == a*xm + b)
m.Minimize((yp-ym)**2)

# solve
m.options.IMODE = 2
m.solve(disp=False)
p1 = [a.value[0],b.value[0]]

print('Slope:', p1[0])
print('Intercept:', p1[1])

# add constraint to the slope (>=0)
a.LOWER = 0.0
m.solve(disp=False)
p2 = [a.value[0],b.value[0]]

print('Slope:', p2[0])
print('Intercept:', p2[1])

# plot results
plt.plot(x,y,'r.')
plt.plot(x,np.polyval(p1,x),label='Unconstrained')
plt.plot(x,np.polyval(p2,x),label='Constrained (slope>=0)')
plt.ylabel('y'); plt.xlabel('x'); plt.legend()
plt.show()