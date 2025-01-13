# load the cancer data from sklearn into X and y
import sklearn.datasets as datasets
X, y = datasets.load_breast_cancer(return_X_y=True)
print("cancer data size is {}".format(X.shape))
print("cancer target size is {}".format(y.shape))

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit the test data with naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# predict the test data
y_est = gnb.predict(X_test)

# generate the confusion matrix
import sklearn.metrics as metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_est)
print('Confusion Matrix:\n {}'.format(confusion_matrix))
