import xgboost as xgb
xgbc = xgb.XGBRegressor()
xgbc.fit(XA,yA)
yP = xgbc.predict(XB)