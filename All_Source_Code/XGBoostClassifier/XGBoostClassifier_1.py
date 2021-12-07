import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(XA,yA)
yP = xgbc.predict(XB)