import numpy as np
from numpy.random import normal
from scipy.stats import chi2

from sklearn.metrics import accuracy_score
from sklearn import tree

N_train, N_test = 2000, 10000
X_train = normal(size=(N_train, 10))
X_test = normal(size=(N_test, 10))

Y_train = np.sum(X_train ** 2, axis=1) < chi2.ppf(0.5, 10)
Y_test = np.sum(X_test ** 2, axis=1) < chi2.ppf(0.5, 10)

G, alpha, M = [], [], 800
w = np.array([1.0/N_train]*N_train)
for i in range(M):
    dtc = tree.DecisionTreeClassifier(max_leaf_nodes=2) # weak predictor
    dtc.fit(X_train, Y_train, sample_weight=w)
    Y_train_hat = dtc.predict(X_train)
    err = np.sum((Y_train_hat != Y_train) * w) / np.sum(w)
    alpha_i = np.log((1-err)/err)
    w = w*np.exp(alpha_i*(Y_train_hat != Y_train))
    G.append(dtc)
    alpha.append(alpha_i)

boosting_iterations_err = []
Y_test_hat = np.zeros(N_test)
for i in range(M):
    tmp = 0+G[i].predict(X_test)
    tmp[tmp == 0] = -1
    Y_test_hat += alpha[i]*tmp
    boosting_iterations_err.append(1-accuracy_score(np.sign(Y_test_hat) == 1, Y_test))

print('Boosting Test Error ', boosting_iterations_err[-1])
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(range(M)), y=boosting_iterations_err))
fig.show()