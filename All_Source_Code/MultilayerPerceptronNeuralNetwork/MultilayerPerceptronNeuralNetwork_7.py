from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt  

# generate training data
x = np.linspace(0.0,2*np.pi)
xr = x.reshape(-1,1)
y = np.sin(x)

# train
nn = MLPRegressor(hidden_layer_sizes=(3), 
                  activation='tanh',\
                  solver='lbfgs',max_iter=2000)
model = nn.fit(xr,y)

# validate
xp = np.linspace(-2*np.pi,4*np.pi,100)
xpr = xp.reshape(-1,1)
yp = nn.predict(xpr)
ypr = yp.reshape(-1,1)
r2 = nn.score(xpr,ypr)
print('R^2: ' + str(r2))

plt.figure()
plt.plot(x,y,'bo')
plt.plot(xpr,ypr,'r-')
plt.show()