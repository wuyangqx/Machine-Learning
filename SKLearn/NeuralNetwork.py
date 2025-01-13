# load a classification dataset from sklearn with 1000 samples, 30 features and random_state=10
import sklearn.datasets as datasets
X, y = datasets.make_classification(n_samples=1000, n_features=30, random_state=10)
print(X.shape)
print(y.shape)

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# fit the training data with a neural network classifier, 
# with batch size 32, hidden layer size 32, solver sgd, learning rate 0.01, max_iter 300, tol 1e-3
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(batch_size=32, 
                    hidden_layer_sizes=(64, 32), 
                    solver='sgd', 
                    shuffle=True, 
                    learning_rate_init=0.001, 
                    max_iter=300, 
                    tol=1e-3)
mlp.fit(train_x, train_y)

# plot the loss curve
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[i for i in range(len(mlp.loss_curve_))], y=mlp.loss_curve_, mode='lines+markers'))
fig.update_layout(title='Loss Curve', xaxis_title='Iteration', yaxis_title='Loss')
fig.show()

