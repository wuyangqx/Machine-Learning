# import sklearn datasets and use it to generate a classification data with 3000 samples, 30 features, 4 classes and 20 informative features
import sklearn.datasets as datasets
X, y = datasets.make_classification(n_samples=3000, n_features=30, n_classes=4, n_informative=20, random_state=42)
print(X.shape)
print(y.shape)

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# define a for loop, with different number of estimators, to fit the training data with RandomForestClassifier, record the training and test scores, and plot the scores
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.graph_objects as go
n_estimators = list(range(1, 41))
train_scores = []
test_scores = []
for n in n_estimators:
    rfc = RandomForestClassifier(n_estimators=n)
    rfc.fit(train_x, train_y)
    train_scores.append(rfc.score(train_x, train_y))
    test_scores.append(rfc.score(test_x, test_y)
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=n_estimators, y=train_scores, mode='lines+markers', name='train scores'))
fig.add_trace(go.Scatter(x=n_estimators, y=test_scores, mode='lines+markers', name='test scores'))
fig.show()