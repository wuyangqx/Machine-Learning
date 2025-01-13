from sklearn.tree import DecisionTreeRegressor
from numpy import exp
import numpy as np


class CustomGradientBoostingClassifier:
    
    def __init__(self, learning_rate, n_estimators, max_depth=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        
        F0 = np.log(y.mean()/(1-y.mean()))  # log-odds values
        self.F0 = np.full(len(y), F0)  # converting to array with the input length
        Fm = self.F0.copy()
        
        for _ in range(self.n_estimators):
            p = np.exp(Fm) / (1 + np.exp(Fm))  # converting back to probabilities
            r = y - p  # residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=0)
            tree.fit(X, r)
            ids = tree.apply(X)  # getting the terminal node IDs

            # looping through the terminal nodes 
            for j in np.unique(ids):
              fltr = ids == j

              # getting gamma using the formula (Σresiduals/Σp(1-p))
              num = r[fltr].sum()
              den = (p[fltr]*(1-p[fltr])).sum()
              gamma = num / den

              # updating the prediction
              Fm[fltr] += self.learning_rate * gamma

              # replacing the prediction value in the tree
              tree.tree_.value[j, 0, 0] = gamma

            self.trees.append(tree)
            
    def predict_proba(self, X):
        
        Fm = self.F0
        
        for i in range(self.n_estimators):
            Fm += self.learning_rate * self.trees[i].predict(X)
            
        return np.exp(Fm) / (1 + np.exp(Fm))  # converting back into probabilities
    

# compare the custom implementation with the sklearn implementation withe RMSE
from sklearn.ensemble import GradientBoostingClassifier

X = np.sort(np.random.rand(100, 1), axis=0)
y = np.random.randint(0, 2, 100)


custom_gbc = CustomGradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=1)
custom_gbc.fit(X, y)
custom_y_pred = custom_gbc.predict_proba(X)

sklearn_gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=1)
sklearn_gbc.fit(X, y)
sklearn_y_pred = sklearn_gbc.predict_proba(X)

from sklearn.metrics import log_loss
custom_log_loss = log_loss(y, custom_y_pred)
sklearn_log_loss = log_loss(y, sklearn_y_pred)
print("Custom Log Loss: ", custom_log_loss)
print("Sklearn Log Loss: ", sklearn_log_loss)