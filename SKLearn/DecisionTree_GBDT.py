# load circles data from sklearn
import sklearn.datasets as datasets
X, y = datasets.make_circles(200, noise=0.2, factor=0.1, random_state=12)
print(X.shape)
print(y.shape)

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# fit the training data with GradientBoostingClassifier with regualarization
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(
    n_estimators=100
)
gbc.fit(train_x, train_y)

# predict the test data
pred_y = gbc.predict(test_x)

# generate the classification report
import sklearn.metrics as metrics
cr = metrics.classification_report(test_y, pred_y)
print(cr)

# plot the decision boundary
import numpy as np
import plotly.graph_objects as go
x0 = np.linspace(-2, 2, 100)
x1 = np.linspace(-2, 2, 100)
X0, X1 = np.meshgrid(x0, x1)
X_grid = np.c_[X0.ravel(), X1.ravel()]
y_grid = gbc.predict(X_grid).reshape(X0.shape)
fig = go.Figure()
fig.add_trace(go.Contour(x=x0, y=x1, z=y_grid, colorscale='Viridis', showscale=False))
fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode='markers', name='class 0'))
fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode='markers', name='class 1'))
fig.show()
