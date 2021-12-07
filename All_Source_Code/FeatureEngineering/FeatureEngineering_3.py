data['T1_mean']  = data['T1'].rolling(window=100,center=True).mean()
data['T1_stdev'] = data['T1'].rolling(window=100,center=True).std()
data['T1_skew']  = data['T1'].rolling(window=100,center=True).skew()
data[['T1','T1_mean']].plot(figsize=(8,4),ylabel='Temperature (degC)')