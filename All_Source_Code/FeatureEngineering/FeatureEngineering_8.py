from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=2, input_type='string')
ht = fh.fit_transform(data['Color']).toarray()
nc = pd.DataFrame(ht)
nc.columns = ['Color_'+str(i) for i in range(fh.n_features)]
data = data.join(nc)
print(data.head())

# create plot to show new features
import matplotlib.pyplot as plt
plt.figure(figsize=(8,3))
plt.rcParams['axes.facecolor'] = 'black'
for i in range(len(data)):
    plt.plot([0,data['Color_0'][i]],\
             [0,data['Color_1'][i]],\
             color=data['Color'][i],\
             marker='o',linestyle='-',lw=3)
plt.show()