from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# The digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split into train and test subsets (50% each)
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# test on second half of data
n = np.random.randint(int(n_samples/2),n_samples)
plt.imshow(digits.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print('Predicted: ' + str(classifier.predict(digits.data[n:n+1])[0]))

# generate confusion matrix
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
plt.show()