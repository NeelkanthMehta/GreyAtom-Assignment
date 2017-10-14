#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Homework Assignment"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import sklearn.datasets as datasets
import pandas as pd
from sklearn.metrics import mean_squared_error

#X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=30)
#regressor = LinearRegression()
#regressor.fit(X_train,y_train)
#print(regressor.intercept_,regressor.coef_)

def splitnfit(x,y,r2=False):
    assert isinstance(x,pd.DataFrame),'Predictor variable(s) dataset must be a pandas.DataFrame'
    assert isinstance(y,pd.DataFrame),'Response variable(s) dataset must be a pandas.DataFrame'
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=30)
    reg = LinearRegression().fit(X_train,y_train)
    return 'Intercept: {0} \n slope coeffs: {1} \n mean sq error: {2} \n r2: {3}'.format(
            reg.intercept_, reg.coef_,mean_squared_error(y_test,reg.predict(X_test))**0.5, reg.score(X_test,y_test))

boston = datasets.load_boston()
boston.keys()
X = boston.data
y = boston.target

pred = pd.DataFrame(X)
resp = pd.DataFrame(y)

splitnfit(pred,resp)