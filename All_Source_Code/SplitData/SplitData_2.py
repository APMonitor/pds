from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True)

print(len(X),len(X_train),len(X_test))