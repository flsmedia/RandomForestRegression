#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:13:29 2017

@author: juan-pablo
"""

# Random Forest Regression


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Even if we only need one column, X must always be a matrix, so we specify a matrix of 10 rows and a column (1:2 the 2 is left out)
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Not need here as the dataset is too small
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Feature Scaling
# Not need here because we are going to use the linearRegression who does the scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,
                                  random_state = 0)
regressor.fit(X,y)


# Predicting a new result with Random Forest Regression
y_pred = regressor.predict(6.5)


# Visualizing the Random Forest Regression results (for higher resolution and smooth curve)
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()