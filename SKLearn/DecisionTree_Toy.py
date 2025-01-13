# load the iris data from sklearn datasets into X and y
import sklearn.datasets as datasets
X, y = datasets.load_iris(return_X_y=True)
print("iris data size is {}".format(X.shape))
print("iris target size is {}".format(y.shape))

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# fit the training data with DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(train_x, train_y)

# predict the test data
pred_y = tree.predict(test_x)

# generate the classification report
import sklearn.metrics as metrics
cr = metrics.classification_report(test_y, pred_y)
print(cr)



