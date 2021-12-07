import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# animate plot?
animate=False

# Load data
file1 = 'http://apmonitor.com/do/uploads/Main/tclab_data5.txt'
file2 = 'http://apmonitor.com/do/uploads/Main/tclab_data6.txt'
data = pd.read_csv(file2)

# Input Features: Temperature and 1st / 2nd Derivatives

# Cubic polynomial fit of temperature using 10 data points
data['dT1'] = np.zeros(len(data))
data['d2T1'] = np.zeros(len(data))
for i in range(len(data)):
    if i<len(data)-10:
        x = data['Time'][i:i+10]-data['Time'][i]
        y = data['T1'][i:i+10]
        p = np.polyfit(x,y,3)
        # evaluate derivatives at mid-point (5 sec)
        t = 5.0
        data['dT1'][i] = 3.0*p[0]*t**2 + 2.0*p[1]*t+p[2]
        data['d2T1'][i] = 6.0*p[0]*t + 2.0*p[1]
    else:
        data['dT1'][i] = np.nan
        data['d2T1'][i] = np.nan

# Remove last 10 values
X = np.array(data[['T1','dT1','d2T1']][0:-10])
y = np.array(data[['Q1']][0:-10])

# Scale data
# Input features (Temperature and 2nd derivative at 5 sec)
s1 = MinMaxScaler(feature_range=(0,1))
Xs = s1.fit_transform(X)
# Output labels (heater On / Off)
ys = [True if y[i]>50.0 else False for i in range(len(y))]

# Split into train and test subsets (50% each)
XA, XB, yA, yB = train_test_split(Xs, ys, \
                    test_size=0.5, shuffle=False)

# Plot regression results
def assess(P):
    plt.figure()
    plt.scatter(XB[P==1,0],XB[P==1,1],marker='^',color='blue',label='True')
    plt.scatter(XB[P==0,0],XB[P==0,1],marker='x',color='red',label='False')
    plt.scatter(XB[P!=yB,0],XB[P!=yB,1],marker='s',color='orange',\
                alpha=0.5,label='Incorrect')
    plt.legend()

# Supervised Classification

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(XA,yA)
yP1 = lr.predict(XB)
#assess(yP1)

# Naïve Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(XA,yA)
yP2 = nb.predict(XB)
#assess(yP2)

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='modified_huber', shuffle=True,random_state=101)
sgd.fit(XA,yA)
yP3 = sgd.predict(XB)
#assess(yP3)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(XA,yA)
yP4 = knn.predict(XB)
#assess(yP4)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=10,random_state=101,\
                               max_features=None,min_samples_leaf=5)
dtree.fit(XA,yA)
yP5 = dtree.predict(XB)
#assess(yP5)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=1,\
                  random_state=101,max_features=None,min_samples_leaf=3)
rfm.fit(XA,yA)
yP6 = rfm.predict(XB)
#assess(yP6)

# Support Vector Classifier
from sklearn.svm import SVC
svm = SVC(gamma='scale', C=1.0, random_state=101)
svm.fit(XA,yA)
yP7 = svm.predict(XB)
#assess(yP7)

# Neural Network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,max_iter=200,\
                    activation='relu',hidden_layer_sizes=(10,30,10),\
                    random_state=1, shuffle=True)
clf.fit(XA,yA)
yP8 = clf.predict(XB)
#assess(yP8)

# Unsupervised Classification

# K-Means Clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(XA)
yP9 = km.predict(XB)
# Arbitrary labels with unsupervised clustering may need to be reversed
#yP9 = 1.0-yP9
#assess(yP9)

# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm.fit(XA)
yP10 = gmm.predict_proba(XB) # produces probabilities
# Arbitrary labels with unsupervised clustering may need to be reversed
yP10 = 1.0-yP10[:,0]

# Spectral Clustering
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=2,eigen_solver='arpack',\
                        affinity='nearest_neighbors')
yP11 = sc.fit_predict(XB) # No separation between fit and predict calls
                        #  need to fit and predict on same dataset
# Arbitrary labels with unsupervised clustering may need to be reversed
#yP11 = 1.0-yP11
#assess(yP11)

plt.figure(figsize=(12,8))

if animate:
    plt.ion()
    plt.show()
    make_gif = True
    try:
        import imageio  # required to make gif animation
    except:
        print('install imageio with "pip install imageio" to make gif')
        make_gif=False
    if make_gif:
        try:
            import os
            images = []
            os.mkdir('./frames')
        except:
            print('Figure directory already created')

gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,5])
plt.subplot(gs[0])
plt.plot(data['Time']/60,data['T1'],'r-',\
         label='Temperature (°C)')
plt.ylabel('T (°C)')
plt.legend(loc=3)
plt.subplot(gs[1])
plt.plot(data['Time']/60,data['dT1'],'b:',\
         label='dT/dt (°C/sec)')    
plt.plot(data['Time']/60,data['d2T1'],'k--',\
         label=r'$d^2T/dt^2$ ($°C^2/sec^2$)')
plt.ylabel('Derivatives')
plt.legend(loc=3)

plt.subplot(gs[2])
plt.plot(data['Time']/60,data['Q1']/100,'k-',\
         label='Heater (On=1/Off=0)')

t2 = data['Time'][len(yA):-10].values
plt.plot(t2/60,yP1-1,label='Logistic Regression')
plt.plot(t2/60,yP2-2,label='Naïve Bayes')
plt.plot(t2/60,yP3-3,label='Stochastic Gradient Descent')
plt.plot(t2/60,yP4-4,label='K-Nearest Neighbors')
plt.plot(t2/60,yP5-5,label='Decision Tree')
plt.plot(t2/60,yP6-6,label='Random Forest')
plt.plot(t2/60,yP7-7,label='Support Vector Classifier')
plt.plot(t2/60,yP8-8,label='Neural Network')
plt.plot(t2/60,yP9-9,label='K-Means Clustering')
plt.plot(t2/60,yP10-10,label='Gaussian Mixture Model')
plt.plot(t2/60,yP11-11,label='Spectral Clustering')

plt.ylabel('Heater')
plt.xlabel(r'Time (min)')
plt.legend(loc=3)

if animate:
    t = data['Time'].values/60
    n = len(t)
    for i in range(60,n+1,10):
        for j in range(3):
            plt.subplot(gs[j])
            plt.xlim([t[max(0,i-1200)],t[i]])        
        filename='./frames/frame_'+str(1000+i)+'.png'
        plt.savefig(filename)
        if make_gif:
            images.append(imageio.imread(filename))
        plt.pause(0.1)

    # create animated GIF
    if make_gif:
        imageio.mimsave('animate.gif', images)
        imageio.mimsave('animate.mp4', images)
else:
    plt.show()