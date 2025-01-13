from sklearn.tree import DecisionTreeRegressor

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.F0 = y.mean()
        
        Fm = self.F0
        for _ in range(self.n_estimators):
            r = y - Fm
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, r)
            Fm += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        Fm = self.F0

        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
        
        return Fm
    
# compare the custom implementation with the sklearn implementation withe RMSE
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

X = np.sort(np.random.rand(100, 1), axis=0)
y = 3*X[:,0] + np.random.randn(100)

custom_gbr = CustomGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1)
custom_gbr.fit(X, y)
custom_y_pred = custom_gbr.predict(X)

sklearn_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1)
sklearn_gbr.fit(X, y)
sklearn_y_pred = sklearn_gbr.predict(X)

from sklearn.metrics import mean_squared_error
custom_rmse = mean_squared_error(y, custom_y_pred)
sklearn_rmse = mean_squared_error(y, sklearn_y_pred)
print("Custom RMSE: ", custom_rmse)
print("Sklearn RMSE: ", sklearn_rmse)

# plot X,y and predictions with graph objects
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=X[:,0], y=y, mode='markers', name='True'))
fig.add_trace(go.Scatter(x=X[:,0], y=custom_y_pred, mode='lines', name='Custom'))
fig.add_trace(go.Scatter(x=X[:,0], y=sklearn_y_pred, mode='lines', name='Sklearn'))
fig.show()



    