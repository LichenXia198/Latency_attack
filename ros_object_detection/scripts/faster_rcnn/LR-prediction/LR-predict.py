#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

def read_file(filename):
	f = open(filename)
	a = []
	line = f.readline()
	while line:
		a.append(float(line.replace('\n', '')))
		line = f.readline()
	f.close()

	return a

box = read_file("box.log")
proposals = read_file("proposals.log")
drawbox = read_file("drawbox.log")

X = [proposals, box]
y = drawbox

X = np.array(X)
Y = np.array(y)
X = np.transpose(X)

print(X.shape)
print(Y.shape)

# Split the data into training/testing sets
X_train = X[:-500]
X_test = X[-500:]

# Split the targets into training/testing sets
y_train = y[:-500]
y_test = y[-500:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

print('median_absolute_error: %.2f' % median_absolute_error(y_test, y_pred))
print('mean value: %.2f' % np.mean(y_test))
accuracy = 1 - median_absolute_error(y_test, y_pred) / np.mean(y_test)
print('average accuracy: %.2f' % accuracy)

# (600, 2)
# (600,)
# Coefficients: 
#  [8.92771912e-04 6.88506573e+00]
# Mean squared error: 0.78
# Coefficient of determination: 1.00
# median_absolute_error: 0.52
# mean value: 31.03
# average accuracy: 0.98