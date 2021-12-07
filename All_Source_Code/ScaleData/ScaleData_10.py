from sklearn.preprocessing import StandardScaler
s = StandardScaler()
sdata = s.fit_transform(data)
sdata = pd.DataFrame(sdata, columns=data.columns.values, index=data.index)
sdata.plot(kind='hist',alpha=0.7,bins=10,figsize=(8,4))