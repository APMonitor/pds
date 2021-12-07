# Create Support Vector Classifier
classifier = svm.SVC(gamma=0.001)
# Learn / Fit
classifier.fit(X_train, y_train)
# Predict
classifier.predict(digits.data[n:n+1])[0]