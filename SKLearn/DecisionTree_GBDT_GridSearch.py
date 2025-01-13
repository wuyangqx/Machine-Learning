# load the cancer data from sklearn into X and y
import sklearn.datasets as datasets
X, y = datasets.load_breast_cancer(return_X_y=True)
print("cancer data size is {}".format(X.shape))
print("cancer target size is {}".format(y.shape))

# split the data into training and test sets, with 80% training and 20% test, random_state=42
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define a GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

# use Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 150],
    'min_samples_leaf': [1, 2, 3]
}
grid_search = GridSearchCV(gbc, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))

# calculate the F1 score with the best parameters
from sklearn.metrics import f1_score
best_gbc = grid_search.best_estimator_
y_est = best_gbc.predict(X_test)
f1 = f1_score(y_test, y_est)
print("F1 score: {}".format(f1))
