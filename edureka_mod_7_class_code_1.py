# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:58:54 2019

@author: 140524
"""

#Practie Edureka class - code1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('C:\\Users\\140524\\Desktop\\D7_Session_Practise\\data.csv')
data.head()
data.columns
data.info()
#dropping unmaned 32 and id columns
data.drop(['Unnamed: 32','id'], axis = 'columns', inplace = True)
data.info()
data.diagnosis.unique()
# encoding diagonsis

data['diagnosis'] = data.diagnosis.map({'M':0,'B':1})
data['diagnosis']

#understanding the data

data.describe()
data.groupby('diagnosis').mean()
sns.countplot(data.diagnosis, label = 'count')
sns.show()

# or

plt.hist(data.diagnosis, label = 'count')
plt.show()

#or

data.diagnosis.value_counts().plot(kind='bar')

# finding correlation

corr = data.corr()
corr

# plotting corr matrix

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
heatMap(data)


# or

corr = data.corr() # .corr is used for find corelation
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar = True,  square = True,
            cmap= 'coolwarm')
plt.show()

# listing all correlated columns

corr_matrix = data.corr()
corr_matrix
corr_matrix.columns[0]
corr_matrix.index[1]
drop_col = []
len(corr_matrix)
for i in range(len(corr_matrix)):
               for j in range(i):
                   if (abs(corr_matrix.iloc[i,j])>=0.75) and (corr_matrix.columns[j] not in drop_col):
                           drop_col.append(corr_matrix.columns[j])
#highly correlated columns to be dropped
drop_col 
#Based on corrplot let's select some features for the model ( decision is made in order to remove collinearity)
prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
# now these are the variables which will use for prediction
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
X = data[prediction_var]
y = data[['diagnosis']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

# AUC/ROC

from sklearn.metrics import confusion_matrix
print(metrics.confusion_matrix(y_test,y_pred))
confusion = metrics.confusion_matrix(y_test,y_pred)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

# =============================================================================
# as u now know classification accuracy can be calculated as: TP+TN/float(TP+TN+FP+FN)
# use float in deno to compute true division
# 
# =============================================================================
print((TP+TN)/float(TP+TN+FP+FN))

# IS EQUAL TO

print(metrics.accuracy_score(y_test,y_pred))

# =============================================================================
#     Classification error aka: Misclassification Rate
#  
# how often the clarrifier is incorrect    
# see below
# =============================================================================

print((FP+FN)/float(TP+TN+FP+FN))

# IS EQUAL TO

print(1 - metrics.accuracy_score(y_test,y_pred))
# =============================================================================
#               Sentivity
# when the actual value is posiive - how often is the prediction correct - TP
# how sensitive is the classifier in detecting those positive instances
# Also known as 'true positve rate' or 'Recall'

#see below               
# =============================================================================

print(TP/float(TP+FN))
# IS EQUAL TO

print(metrics.recall_score(y_test,y_pred))
# =============================================================================
#               Specificty
# when the actual value is negaiive - how often is the prediction correct - TN
# how specific is the classifier in detecting those negative instances


#see below               
# =============================================================================

print(TN/float(TN+FP))

# False positive rate: when the actual value is negaiive - how often is the prediction incorrect - FN
#False posiive rate is 1 - specificity

print(FP/float(TN+FP))

#Precision: what percentage of predictions are correct considering the total 
#number of positive 
#predictions made
print(TP/float(FP+TP))

# ie equal to

print(metrics.precision_score(y_test,y_pred))

#print the first 10 predicted response

log.predict(X_test)[0:10]

#print the first 10 predicted probabilities of class membership
log.predict_proba(X_test)[0:10,:]  

log.predict_proba(X_test)[0:10,1]      

y_pred_prob = log.predict_proba(X_test)[:,1] 
 
import matplotlib.pyplot as plt
#plt.rcParams['font.size']=14
plt.hist(y_pred,bins=8)
plt.xlim(0,1)
plt.title('histogram of predicted probability')
plt.xlabel('predicted probability of diabetes') 
plt.ylabel('frequency')    

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_pred,y_test)
roc_auc = auc(fpr, tpr)
roc_auc

fpr,tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False positive rate(1-specificity)')
plt.ylabel('True positive rate(sentisivity)')

def evaluate_threshold(threshold):
    print('sensitivity:',tpr[thresholds>threshold[-1]])
    print('specificity:',1-fpr[thresholds>threshold[-1]])
    
evaluate_threshold(1/2)



import numpy as np    
from sklearn.preprocessing import binarize
# has to convert it to a 2D array because binarize function expects it to be
#done by reshaping
y_pred_class = binarize(y_pred.reshape(1,-1),0.3)
y_pred_prob[0:10] 
y_pred_prob[0:10] 
#now converting it back to a 1D array by ravel because confusion matrix wants to be 1D array
y_pred_class1 = y_pred_class.ravel()
y_pred_class1

type(y_pred_class1)



