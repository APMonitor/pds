select_option = 3

data_options = ['linear','quadratic','target','moons','circles','blobs']
option = data_options[select_option]
n = 2000 # number of data points

X = np.random.random((n,2))
mixing = 0.0 # add random mixing element to data
xplot = np.linspace(0,1,100)

if option=='linear':
    y = np.array([False if (X[i,0]+X[i,1])>=(1.0+mixing/2-np.random.rand()*mixing) \
                    else True \
                  for i in range(n)])
    yplot = 1-xplot
elif option=='quadratic':
    y = np.array([False if X[i,0]**2>=X[i,1]+(np.random.rand()-0.5)*mixing \
                    else True \
                  for i in range(n)])
    yplot = xplot**2
elif option=='target':
    y = np.array([False if (X[i,0]-0.5)**2+(X[i,1]-0.5)**2<=0.1 +(np.random.rand()-0.5)*0.2*mixing \
                    else True \
                  for i in range(n)])
    j = False
    yplot = np.empty(100)
    for i,x in enumerate(xplot):
        r = 0.1-(x-0.5)**2
        if r<=0:
            yplot[i] = np.nan
        else:
            j = not j # plot both sides of circle
            yplot[i] = (2*j-1)*np.sqrt(r)+0.5
elif option=='moons':
    X, y = datasets.make_moons(n_samples=n,noise=0.05)
    yplot = xplot*0.0
elif option=='circles':
    X, y = datasets.make_circles(n_samples=n,noise=0.05,factor=0.5)
    yplot = xplot*0.0
elif option=='blobs':
    X, y = datasets.make_blobs(n_samples=n,centers=[[-5,3],[5,-3]],cluster_std=2.0)
    yplot = xplot*0.0

plt.scatter(X[y>0.5,0],X[y>0.5,1],color='blue',marker='^',label='True')
plt.scatter(X[y<0.5,0],X[y<0.5,1],color='red',marker='x',label='False')
if option not in ['moons','circles','blobs']:
    plt.plot(xplot,yplot,'k.',label='Division')
plt.legend()

# Split into train and test subsets (50% each)
XA, XB, yA, yB = train_test_split(X, y, test_size=0.5, shuffle=False)

# Plot regression results
def assess(P):
    plt.figure()
    plt.scatter(XB[P==1,0],XB[P==1,1],marker='^',color='blue',label='True')
    plt.scatter(XB[P==0,0],XB[P==0,1],marker='x',color='red',label='False')
    plt.scatter(XB[P!=yB,0],XB[P!=yB,1],marker='s',color='orange',alpha=0.5,label='Incorrect')
    if option not in ['moons','circles','blobs']:
        plt.plot(xplot,yplot,'k.',label='Division')
    plt.legend()