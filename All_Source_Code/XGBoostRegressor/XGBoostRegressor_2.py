import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

X, y = make_regression(n_samples=1000, n_features=10, n_informative=8)
Xa,Xb,ya,yb = train_test_split(X, y, test_size=0.2, shuffle=True)
xgbr = xgb.XGBRegressor()
xgbr.fit(Xa,ya)
yp = xgbr.predict(Xb)
acc = r2_score(yb,yp)
print('R2='+str(acc))

# importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
f = xgbr.get_booster().get_score(importance_type='gain')
t = pd.DataFrame(f.items(),columns=['Feature','Gain'])
print(t.sort_values('Gain',ascending=False))

# create table from feature_importances_
fi = xgbr.feature_importances_
n = ['Feature '+str(i) for i in range(10)]
d = pd.DataFrame({'Feature':n,'Importance':fi})
print(d.sort_values('Importance',ascending=False))

# plot the importance by 'weight'
xgb.plot_importance(xgbr)