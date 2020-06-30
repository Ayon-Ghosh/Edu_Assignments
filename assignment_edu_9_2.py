# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 01:15:58 2019

@author: 140524
"""

#Module 9
#case_study 2

import pandas as pd
df = pd.read_csv('C:\\edureka\\PR_Class_9\\Data_set\\run_or_walk.csv')
df.head()
df.columns
df.drop(['date','time','username'], axis=1, inplace = True)
df.columns
X = df.drop('activity', axis=1)
X.columns
X.shape
y = df[['activity']]
#Test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Training Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(y_test)
# creating confusion matrix
from sklearn import metrics
cnf = metrics.confusion_matrix(y_test,y_pred)
print(cnf)
# measuring score
print(metrics.accuracy_score(y_test,y_pred))

# missclassification error

print(1-metrics.accuracy_score(y_test,y_pred))

print('Number of mislabeled items out of a total of %d items: %d'%(df.shape[0],(y_test!=y_pred).sum()))


#using acceleration as X

df.columns
X_new = df[['acceleration_x','acceleration_y','acceleration_z']]
X_new.shape
y = df[['activity']]
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.2)

# Training Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(y_test)
# creating confusion matrix
from sklearn import metrics
cnf = metrics.confusion_matrix(y_test,y_pred)
print(cnf)
# measuring score
print(metrics.accuracy_score(y_test,y_pred))

#using gyro as X

df.columns
X_gyro = df[['gyro_x','gyro_y','gyro_z']]
X_gyro.shape
y = df[['activity']]
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_gyro,y,test_size=0.2)

# Training Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(y_test)
# creating confusion matrix
from sklearn import metrics
cnf = metrics.confusion_matrix(y_test,y_pred)
print(cnf)
# measuring score
print(metrics.accuracy_score(y_test,y_pred))


# Comment on the difference in accuracy between both the models.

#Using acceleration, the True postive is very high which means the accuracy on 
#prediction of activity 1 is very high and the accuracy of prediction on activuty 
# 0 is very low

# Using gyro - its vice versa - the true negative which means the accuracy on 
#prediction of activity 0 is very high and the accuracy of prediction on activuty 
# 1 is very low