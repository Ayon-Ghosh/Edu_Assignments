# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:37:20 2019

@author: 140524
"""

#Module 7
#case study 2
#1

import pandas as pd
horse = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\horse.csv')
horse.shape
horse.columns
horse.isnull().sum()


#3
#using for loop and mode()
for i in range(28):
    if (horse[horse.columns[i]].isnull().sum())>0:
                    horse[horse.columns[i]].fillna(horse[horse.columns[i]].mode().iloc[0], inplace = True)
                    
horse.head(10)
horse.isnull().sum()

# using Imputer class
#https://scikit-learn.org/stable/modules/impute.html
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
horse1 = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\horse.csv')
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(horse1)
print(pd.DataFrame(imp.transform(horse1)))

# 2 -- Get_Dummy


horse.dtypes
horse_cat = horse.select_dtypes(exclude='number').head()
horse_cat
horse_num = horse.select_dtypes(exclude='object').head()
horse_num
horse_cat = pd.get_dummies(horse_cat, drop_first=True)
horse_cat.columns

# Decision Tree Classifier

X = pd.concat([horse_cat,horse_num], axis = 'columns')
X.columns
X.shape

X = pd.concat([horse_cat,horse_num], axis = 'columns').drop(['outcome_lived','outcome_euthanized'],axis='columns')
X.columns
X.shape
y = horse_cat[['outcome_lived','outcome_euthanized']]
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# using label_encoder

horse.dtypes
cat_cols = [i for i in horse.columns if horse[i].dtype == 'object']
cat_cols
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_cols:
      horse[i] = le.fit_transform(horse[i])
X = horse.drop(['outcome'], axis = 'columns')
y = horse.outcome
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)

# compute the feature importances
pd.DataFrame({'feature':X.columns, 'importance':model.feature_importances_})

# Using RandomForest
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
model.score(X_test,y_test)