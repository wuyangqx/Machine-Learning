import numpy as np

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        # the training step is essentially storing the training data, as KNN is memory-based model free algorithm
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
class KNNRegressor:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the mean of the k nearest neighbor training samples
        return np.mean(k_nearest_labels)


# load the iris dataset as X and y
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Test KNNClassifier
clf = KNNClassifier(k=3)
clf.fit(X, y)
predictions = clf.predict(X)
print(y)
print(predictions)

# Train SKLearn KNN Classifier with X and y
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)
predictions = clf.predict(X)
print(y)
print(predictions)


# # Test KNNRegressor
# reg = KNNRegressor(k=3)
# reg.fit(X, y)
# predictions = reg.predict(X)
# print(y)
# print(predictions)

# # Train SKLearn KNN Regressor with X and y
# from sklearn.neighbors import KNeighborsRegressor
# reg = KNeighborsRegressor(n_neighbors=3)
# reg.fit(X, y)
# predictions = reg.predict(X)
# print(y)
# print(predictions)





