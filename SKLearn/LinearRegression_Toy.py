import numpy as np
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5]

# reshape x to be a column vector
lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y)
y_est = lr.predict(x.reshape(-1, 1))
y_err = x.std() * np.sqrt(1 / len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

print('Coefficients: {}'.format(lr.coef_))
print('Bias term: {}'.format(lr.intercept_))
print('Mean squared error: {}'.format(np.mean((y - y_est)**2)))
print('Coefficient of determination: {}'.format(lr.score(x.reshape(-1, 1), y)))

import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data'))
fig.add_trace(go.Scatter(x=x, y=y_est, mode='lines', name='fit'))
fig.add_trace(go.Scatter(x=x, y=y_est, mode='markers', name='fitStd', error_y=dict(type='data', array=y_err, visible=True)))
fig.show()






