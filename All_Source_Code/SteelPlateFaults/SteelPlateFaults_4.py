# Scale data
s = MinMaxScaler()
data_s = s.fit_transform(data)
data_s = pd.DataFrame(data_s,columns=data.columns)

# Split data into X and y
features = data.columns[:-7]
labels = data.columns[-7:]
X = data_s[features]
y = data_s[labels]

# Train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,shuffle=True)