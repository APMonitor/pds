# Retrieve model and test data
import pickle
[classifier,digits] = pickle.load(open('store.pkl','rb'))

# Test on second to last number
print('Predicted: ' + str(classifier.predict(digits.data[-2:-1])[0]))

# Show number
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()