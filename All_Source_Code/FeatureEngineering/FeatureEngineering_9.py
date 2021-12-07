from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
bestfeatures = SelectKBest(score_func=chi2, k='all')
features = ['Brown','Gray','Orange','Tan','White','CNumber']
X = data[features]
y = data['Label']
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
scores = pd.concat([dfcolumns,dfscores],axis=1)
scores.columns = ['Specs','Score']
scores.index = features
scores.plot(kind='bar',figsize=(8,2))
plt.show()