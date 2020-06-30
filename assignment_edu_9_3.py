# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:47:13 2019

@author: 140524
"""

#Module 9
# case study 3

#1
import pandas as pd
college = pd.read_csv('C:\\edureka\\PR_Class_9\\Data_set\\College.csv')
college.head()
college.columns

#2

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
college.Private = le.fit_transform(college.Private)
college.head()

# Understanding data
college.groupby('Private').mean()

#understanding the difference in mneasurements of spread
# for each features for each response category

college1 = college.loc[0]
college1
college1.plot(kind='bar', width = 0.7, color = 'red')
college2 = college.loc[1]
college2
college2.plot(kind='bar' , width = 0.3, color = 'blue')

# from this above graph we can see that the features - 
# Apps, Accept, Enroll,p.undergrad, outstate, Room board, books, personal, Expand
# have separate spreads of measure to impact the response/ outcome catefory

import matplotlib.pyplot as plt
plt.ylim(0,1000)

# from the above we know that the features - Enroll, Top10perc, Top25perc, pdh, 
#terminal, S,F ratio, perc alumni are good predictors

# from above 2 graphs we will choose the features by dropping

# Train Test process
X = college.drop(['Private','F.Undergrad','Grad.Rate'], axis=1)
X.columns

y = college.Private

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)


# 3
# Using test train
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import  metrics
C_range = range(1,100)
score = []
for i in C_range:
  
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)  
  model = LinearSVC(C = i)
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  score.append(metrics.accuracy_score(y_test,y_pred))
dict(zip(C_range,score))
#now plotting the relationship between C and testing accuracy

import matplotlib.pyplot as plt
plt.plot(C_range,score)
plt.xlabel('Value of C')
plt.ylabel('Accuracy')
plt.show()  

#  The best accuracy of 0.97% is achieced with C = 55

# 4--
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics
C_range = range(1,100)
score = []
for i in C_range:
  
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 
  X_train = StandardScaler().fit_transform(X_train)
  model = LinearSVC(C = i)
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  score.append(metrics.accuracy_score(y_test,y_pred))
dict(zip(C_range,score))
#now plotting the relationship between C and testing accuracy

import matplotlib.pyplot as plt
plt.plot(C_range,score)
plt.xlabel('Value of C')
plt.ylabel('Accuracy')
plt.show() 

# The overall accuracy has reduced with fiiting and transforming through Standard_Scaler

# 5
import pandas as pd
college = pd.read_csv('C:\\edureka\\PR_Class_9\\Data_set\\College.csv')
college.head()
college.columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
college.Private = le.fit_transform(college.Private)
college.head()
X = college.drop(['Private','F.Undergrad','Grad.Rate'], axis=1)
X.columns

y = college.Private
from sklearn.model_selection import GridSearchCV
import numpy as np
param_grid=dict(C = np.arange( 1, 100+1, 1 ).tolist(), kernel = ['poly','rbf']  , 
                gamma = np.arange( 0.0, 10.0+0.0, 0.1 ).tolist())
#print(param_grid)
from sklearn.svm import SVC
model = SVC(param_grid)
grid = GridSearchCV(model,param_grid,cv=10,scoring='accuracy')
#fitting the model with data
grid.fit(X,y)
results = pd.DataFrame(grid.cv_results_)
results
results.columns

results[['mean_test_score', 'std_test_score', 'params']]
# examine the results of the first set of parameters
results['params'][0]
#or
results.loc[0,'params']
results['mean_test_score'][0]
# list all of the mean test scores
results['mean_test_score']

# if the std_test_score which is the standard deviation of each of the meam score is 
#high it means that the cross validation accuracy is low

#create a list of mean sores only
#plot the results
C_range = range(1,101)
import matplotlib.pyplot as plt
plt.plot(C_range,results['mean_test_score'])
plt.xlabel('C value')
plt.ylabel('mean cross validation score for each SVM')
plt.show()

#examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

