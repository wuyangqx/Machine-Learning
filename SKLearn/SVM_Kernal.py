# import dataset from sklearn
import sklearn.datasets as datasets

# load the circles data
X, y = datasets.make_circles(200, noise=0.2, factor=0.1, random_state=12)

# print the shape of X and y
print(X.shape)
print(y.shape)

# plot the data using plotly
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode='markers', name='class 0'))
fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode='markers', name='class 1'))
fig.show()

# fit the data with SVM with a radial basis function kernel
from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X, y)

# plot the decision boundary
import numpy as np
x0 = np.linspace(-2, 2, 100)
x1 = np.linspace(-2, 2, 100)
X0, X1 = np.meshgrid(x0, x1)
X_grid = np.c_[X0.ravel(), X1.ravel()]
y_grid = svc.decision_function(X_grid).reshape(X0.shape)
fig = go.Figure()
fig.add_trace(go.Contour(x=x0, y=x1, z=y_grid, colorscale='Viridis', showscale=False))
fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode='markers', name='class 0'))
fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode='markers', name='class 1'))
fig.show()







