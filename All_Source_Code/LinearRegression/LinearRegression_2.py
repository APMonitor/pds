import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
import matplotlib.pyplot as plt
from gekko import GEKKO

# Data
x = np.array([4,5,2,3,-1,1,6,7])
y = np.array([0.3,0.8,-0.05,0.1,-0.8,-0.5,0.5,0.65])

# calculate R^2
def rsq(y1,y2):
    yresid= y1 - y2
    SSresid = np.sum(yresid**2)
    SStotal = len(y1) * np.var(y1)
    r2 = 1 - SSresid/SStotal
    return r2

# Method 1: scipy linregress
slope,intercept,r,p_value,std_err = linregress(x,y)
a = [slope,intercept]
print('R^2 linregress = '+str(r**2))

# Method 2: numpy polyfit (1=linear)
a = np.polyfit(x,y,1); print(a)
yfit = np.polyval(a,x)
print('R^2 polyfit    = '+str(rsq(y,yfit)))

# Method 3: numpy linalg solution
#       y =     X a
#   X^T y = X^T X a
X = np.vstack((x,np.ones(len(x)))).T
# matrix operations
XX = np.dot(X.T,X)
XTy = np.dot(X.T,y)
a = np.linalg.solve(XX,XTy)
# same solution with lstsq
a = np.linalg.lstsq(X,y,rcond=None)[0]
yfit = a[0]*x+a[1]; print(a)
print('R^2 matrix     = '+str(rsq(y,yfit)))

# Method 4: statsmodels ordinary least squares
X = sm.add_constant(x,prepend=False)
model = sm.OLS(y,X).fit()
yfit = model.predict(X)
a = model.params
print(model.summary())

# Method 5: Gekko for constrained regression
m = GEKKO(remote=False); m.options.IMODE=2
c  = m.Array(m.FV,2); c[0].STATUS=1; c[1].STATUS=1
c[1].lower=-0.5
xd = m.Param(x); yd = m.Param(y); yp = m.Var()
m.Equation(yp==c[0]*xd+c[1])
m.Minimize((yd-yp)**2)
m.solve(disp=False)
c = [c[0].value[0],c[1].value[1]]
print(c)

# plot data and regressed line
plt.plot(x,y,'ko',label='data')
xp = np.linspace(-2,8,100)
slope     = str(np.round(a[0],2))
intercept = str(np.round(a[1],2))
eqn = 'LstSQ: y='+slope+'x'+intercept
plt.plot(xp,a[0]*xp+a[1],'r-',label=eqn)
slope     = str(np.round(c[0],2))
intercept = str(np.round(c[1],2))
eqn = 'Constraint: y='+slope+'x'+intercept
plt.plot(xp,c[0]*xp+c[1],'b--',label=eqn)
plt.grid()
plt.legend()
plt.show()