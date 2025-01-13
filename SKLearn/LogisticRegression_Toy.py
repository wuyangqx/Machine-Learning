import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import plotly.graph_objects as go
X, y = datasets.make_classification(n_features=2,
                                    n_redundant=0,
                                    n_informative=1,
                                    n_clusters_per_class=1,
                                    random_state=42)
lr2 = LogisticRegression(solver="liblinear")
lr2.fit(X, y)
line_bias = lr2.intercept_
line_w = lr2.coef_.T

# Create plotly figure
fig = go.Figure()

# Add scatter plot for data points
fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, symbol='circle')))

# Calculate decision boundary
points_x = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
points_y = (line_w[0][0] * points_x + line_bias[0]) / (-1 * line_w[1][0])

# Add decision boundary to plot
fig.add_trace(go.Scatter(x=points_x, y=points_y, mode='lines', name='Decision Boundary'))

# Show plot
fig.show()



