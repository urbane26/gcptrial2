# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:47:53 2022

@author: chakr
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


pd.set_option('display.max_columns', None)

df = pd.read_csv('customdata1.csv')
df = df.astype({'age': 'int64','sex': 'object','cp':'object','fbs': 'object','restecg':'object','exang':'object','slop':'object','ca':'object','thal':'object','num':'object'})

y = df['num']
X = df.drop(['num'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

t = [('imp',SimpleImputer(),[10,11]),('cat',OneHotEncoder(),[1,2,5,6,10,11,12]),('scale',MinMaxScaler(),[0,3,4,7,9])]
transformer = ColumnTransformer(transformers=t,remainder='passthrough')

model = xgb.XGBClassifier(booster = 'gbtree',objective = 'multi:softmax',num_class = 5)

pipeline = Pipeline(steps= [('t',transformer),('m',model)])
pipeline.fit(X_train,y_train)
yhat = pipeline.predict(X_test)

