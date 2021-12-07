import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, n_informative=8)
Xa,Xb,ya,yb = train_test_split(X, y, test_size=0.2, shuffle=True)
xgbr = xgb.XGBRegressor()
xgbr.fit(Xa,ya)
yp = xgbr.predict(Xb)
acc = r2_score(yb,yp)
print(acc)
xgb.plot_importance(xgbr)