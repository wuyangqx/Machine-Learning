import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

fileDir = './Kaggle/netflix-prize-data/'

# read reading data from CSV
fileName = 'combined_data_1.txt'
df = pd.read_csv(fileDir + fileName, header = None, names = ['Cust_Id', 'Rating'], usecols=[0, 1])
dfM = df[df['Rating'].isna()].copy()
dfM.rename(columns = {'Cust_Id' : 'Movie_Id'}, inplace=True)
dfM = dfM.drop(columns=['Rating'])
dfT = pd.merge(left = df, right = dfM, how = 'left', left_index=True, right_index=True)
dfT['Movie_Id'] = dfT['Movie_Id'].ffill().apply(lambda i: i[:-1])
dfT = dfT[~dfT['Rating'].isna()]
dfRating = dfT

# read title data from CSV
fileName = 'movie_titles.csv'
dfTitle = pd.read_csv(fileDir + fileName, encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'], on_bad_lines='skip')

# Basic statistics
movie_count = dfRating['Movie_Id'].nunique()
cust_count = dfRating['Cust_Id'].nunique()
rating_count = len(dfRating['Rating'])
print(f"movie_count {movie_count:,}, cust_count {cust_count:,} rating_count {rating_count:,}")
print(f"movie-user-matrix size = {movie_count * cust_count:,} number of non-zero entries {rating_count:,} percentage of non-zero {rating_count / (movie_count * cust_count):.2%}")


nrows = 100000
df = dfRating.iloc[:nrows].copy()
#df = dfRating.copy()

# import suprise library, load the data from pandas dataframe
from surprise import Reader, Dataset
reader = Reader()
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:], reader)

# split the data into training and testing
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=0.25)

# train the SVD model with the training data
from surprise import SVD
algo = SVD()
algo.fit(trainset)

# make predictions on the test data and calculate the accuracy
predictions = algo.test(testset)
from surprise import accuracy
accuracy.rmse(predictions)

# train KNNBasic model with the training data
from surprise import KNNBasic
algo = KNNBasic()
algo.fit(trainset)

# import SKLearn library, load the data from pandas dataframe into X and y
from sklearn.model_selection import train_test_split
X = df[['Cust_Id', 'Movie_Id']]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# import SVD from SKLearn library, train the model with the training data, and calculate the root mean square error
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
svd = TruncatedSVD(n_components=1)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)
svd.fit(X_train)

# use svd to predict the test data and calculate the root mean square error
y_pred = svd.fit_transform(X_test)
mean_squared_error(y_test, y_pred)





# from surprise import Reader, Dataset
# from surprise.model_selection import cross_validate
# # get just top 100K rows for faster run time
# df = dfRating.iloc[:1000].copy()

# reader = Reader()
# data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:], reader)
# data.split(n_folds=3)
# svd = SVD()
# cross_validate(svd, data, measures=['RMSE', 'MAE'])


# from surprise import SVD
# from surprise.model_selection import train_test_split
# from surprise import accuracy
# algo = SVD()
# reader = Reader()
# data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:], reader)
# trainset, testset = train_test_split(data, test_size=0.25)
# algo.fit(trainset)
# predictions = algo.test(testset)
# accuracy.rmse(predictions)
