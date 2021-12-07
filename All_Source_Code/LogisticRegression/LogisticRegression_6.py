l_rate = 0.3
n_epoch = 100

loss = np.zeros(n_epoch)
beta = [0.0,0.0,0.0]
for epoch in range(n_epoch):
    sum_error = 0
    for row in train:
        x = row[0:-1] # input features
        y = row[-1]   # output label
        yhat = predict(row, beta)
        error = y - yhat
        sum_error += error**2
        beta[0] += l_rate * error * yhat * (1.0 - yhat)
        beta[1] += l_rate * error * yhat * (1.0 - yhat) * x[0]
        beta[2] += l_rate * error * yhat * (1.0 - yhat) * x[1]
    loss[epoch] = sum_error

print('Coefficients:',beta)
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')