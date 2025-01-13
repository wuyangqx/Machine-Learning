# suprise is for recommendation
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

data = Dataset.load_builtin('ml-100k')

algo = SVD()

# simple train/test split
trainset, testset = train_test_split(data, test_size=0.25)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# cross validation
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
