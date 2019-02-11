'''
Created on Aug 14, 2018

@author: abhinav.jhanwar
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import StandardScaler

url = "Housing.csv"
data = pd.read_csv(url)


feature_cols = data.columns.values.tolist()[:-1]
X = data[feature_cols]
y = data[data.columns.values.tolist()[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# rmse
print("RMSE (ERROR IN PREDICTION: Preferred value: <10): ", np.sqrt(mean_squared_error(y_test, y_pred)))

''' APPLYING PCA '''

# standardize x before applying pca
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)

# Create PCA object 
# define number of components after reduction
n_components=10
pca = PCA(n_components=n_components)

# Fit and Apply dimensionality reduction on X
# apply pca only on training data and then fit the same on testing and cv data
X_train_pc = pca.fit_transform(X_train_std)

#The amount of variance that each PC explains
var = pca.explained_variance_ratio_
# find total variance explained by the given number of components
# take total number of components which can explain 99% variance
total_variance = sum(var)
print("Total variance explained by {0} features is: {1}".format(n_components, round(total_variance*100,2)))

model = LinearRegression()
model.fit(X_train_pc, y_train)

X_test_std = scaler.transform(X_test)
X_test_pc = pca.transform(X_test_std)
y_pred = model.predict(X_test_pc)

# rmse
print("RMSE (ERROR IN PREDICTION: Preferred value: <10): ", np.sqrt(mean_squared_error(y_test, y_pred)))
