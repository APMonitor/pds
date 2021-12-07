import numpy as np
import statsmodels.api as sm
from gekko import GEKKO

# Data
x0 = np.array([4,5,2,3,-1,1,6,7])
x1 = np.array([3,2,3,4, 3,5,2,6])
y = np.array([0.3,0.8,-0.05,0.1,-0.8,-0.5,0.5,0.65])

# calculate R^2
def rsq(y1,y2):
    yresid= y1 - y2
    SSresid = np.sum(yresid**2)
    SStotal = len(y1) * np.var(y1)
    r2 = 1 - SSresid/SStotal
    return r2

# Method 1: numpy linalg solution
#       Y =     X a
#   X^T Y = X^T X a
X = np.vstack((x0,x1,np.ones(len(x0)))).T
a = np.linalg.lstsq(X,y)[0]; print(a)
yfit = a[0]*x0+a[1]*x1+a[2]
print('R^2 = '+str(rsq(y,yfit)))

# Method 2: statsmodels ordinary least squares
model = sm.OLS(y,X).fit()
predictions = model.predict(X)
print(model.summary())

# Method 3: gekko
m = GEKKO(remote=False); m.options.IMODE=2
c  = m.Array(m.FV,3)
for ci in c:
    ci.STATUS=1
xd = m.Array(m.Param,2); xd[0].value=x0; xd[1].value=x1
yd = m.Param(y); yp = m.Var()
s =  m.sum([c[i]*xd[i] for i in range(2)])
m.Equation(yp==s+c[-1])
m.Minimize((yd-yp)**2)
m.solve(disp=False)
a = [c[i].value[0] for i in range(3)]
print(a)

# plot data
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
fig = plt.figure()
ax  = plt.axes(projection='3d')
ax.plot3D(x0,x1,y,'ko')
x0t = np.arange(-1,7,0.25)
x1t = np.arange(2,6,0.25)
X0,X1 = np.meshgrid(x0t,x1t)
Yt = a[0]*X0+a[1]*X1+a[2]
ax.plot_surface(X0,X1,Yt,cmap=cm.coolwarm,alpha=0.5)
plt.show()