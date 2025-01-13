import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

# Sample data
data = {
    'Country': ['England', 'Wales', 'Scotland', 'N. Ireland'],
    'Fresh potatoes': [84.0, 78.0, 87.0, 157.0],
    'Fresh veg': [253.0, 265.0, 171.0, 143.0],
    'Fresh fruits': [1153.0, 1102.0, 957.0, 674.0],
    'Cheese': [105.0, 103.0, 103.0, 66.0],
    'Fish': [147.0, 160.0, 122.0, 93.0],
    'Alcoholic drinks': [1193.0, 1312.0, 1389.0, 740.0]
}
df = pd.DataFrame(data)
df.set_index('Country', inplace=True)

n_components = 2

# Perform PCA on the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
pca = PCA(n_components=n_components)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
print("PCA tranformed data:", pca_data)
print(pca.explained_variance_ratio_)

# Perform SVD on the data without scaling
svd = TruncatedSVD(n_components=n_components)
svd.fit(df)
svd_data = svd.transform(df)
print("SVD tranformed data:", svd_data)
print(svd.explained_variance_ratio_)

# Perform factor analysis on the data
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=n_components)
fa.fit(scaled_data)
fa_data = fa.transform(scaled_data)
print("Factor Analysis tranformed data:", fa_data)
#print(fa.explained_variance_ratio_)