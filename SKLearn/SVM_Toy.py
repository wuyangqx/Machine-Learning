
# load cancer data from sklearn
import sklearn.datasets as datasets
from sklearn.svm import SVC
import numpy as np
import plotly.graph_objects as go
X, y = datasets.load_breast_cancer(return_X_y=True)
# svc = SVC(kernel="linear")
# svc.fit(X, y)
# line_bias = svc.intercept_
# line_w = svc.coef_.T

# print X and y data shape
print(X.shape)
print(y.shape)

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit the training data with linear SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

# predict the test data
y_est = svc.predict(X_test)

# generate the confusion matrix
import sklearn.metrics as metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_est)
print('Confusion Matrix:\n {}'.format(confusion_matrix))

# generate the classification report
classification_report = metrics.classification_report(y_test, y_est)
print('Classification Report:\n {}'.format(classification_report))
