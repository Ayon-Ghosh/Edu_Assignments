# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:37:28 2019

@author: 140524
"""


#Module 7
#case study 3

import pandas as pd
df = pd.read_csv('C:\\edureka\\PRClass_7\\Data_set\\loan_borowwer_data.csv')
df.head()
df.dtypes
#checking for null values
df.isnull().sum()
#Label encoding 
df.purpose.unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'])
df.head()
#checking the mean of each features by outcome
df.groupby('not.fully.paid').mean()

#trying to find the correlation
import matplotlib.pyplot as plt
import seaborn as sns
def heatMap(df):
    #Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    #Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()
heatMap(df)
#corr matrix
df.corr()

#Invoking different ML model and measuring the scores indivitually
#Trying our dimensionality reduction (without PCA) and also Parameter Tuning

# dropping the negatively correlated and un correlated features

# Log Reg Model 1
X  = df.drop(['credit.policy','log.annual.inc','fico','days.with.cr.line'], axis='columns')
y = df['not.fully.paid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)

# Log Reg Model 2

X  = df[['credit.policy','log.annual.inc','fico','days.with.cr.line']]
y = df['not.fully.paid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)

#Log Reg Model 3

X  = df.drop('not.fully.paid', axis='columns')
y = df['not.fully.paid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)

#Desicison Tree
X  = df.drop('not.fully.paid', axis='columns')
y = df['not.fully.paid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)

#Random_Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
model.score(X_test,y_test)

# KNN Neighbors

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
k_range=range(1,26)
score = []
for k in k_range:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
#thelist prints all the value of accuracy for each K
score    
dict(zip(k_range,score))
#now plotting the relationship between K and testing accuracy

import matplotlib.pyplot as plt
plt.plot(k_range,score)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()

