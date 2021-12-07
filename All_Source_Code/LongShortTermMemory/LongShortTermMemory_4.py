# Fit LSTM model
history = m.fit(xin, next_X, epochs = 50, batch_size = 50,verbose=0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])