# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:41:52 2019

@author: 140524
"""

#Module 8
# case study 1

#question 1 and 2

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
#each data is an one Dimensional array of numbers. The total array of data[0] 
#represents 1 image. and so on
digits.data[0]
#printing the image
plt.gray()
plt.matshow(digits.images[0])

#printing the first 5 images

for i in range(5):
    plt.matshow(digits.images[i])
#the target gives the identify of the image   
digits.target[0:5]
#each data is an array iof numbers representing 1 hand written image 
digits.data[0:5]
digits.target_names[0:5]
    
#we can use the data and target to built the model

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2)
len(X_train)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
logreg.score(X_test,y_test)

#question 5

# question 5

import matplotlib.pyplot as plt
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
print(cm)

# or using seaborn
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# classification error
from sklearn import metrics
print(1 - metrics.accuracy_score(y_test,y_pred))

# number of instances that has gone wrong

count = 0
for i in range(10):
    for j in range(10):
         if i!=j:
             if cm[i][j]!=0:
                 count = count+1
count    

# for each wrong sample plot the digit, along with predicted and orginal level


for i in range(10):
    for j in range(10):
         if i!=j:
             if cm[i][j]!=0:
                 print('Actual sample:', i)
                 print('predicted sample: ', j)
                 print('#####')
                 plt.matshow(digits.images[i])   
                 print('#####')



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
# question3 and 4

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
#finding the length of each row
l = len(digits.data[0])
#finding the # of rows in the new dataframe
digits.data.shape
#creating the dataframe by appedning rows
import pandas as pd
import numpy as np
df = pd.DataFrame(columns = [str(x) for x in range(0,64)])
df

for i in range(1797):
   df.loc[i] = digits.data[i]
df.head(5)
type(df)
df.shape

# center and scale the date

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Standardizing the features
df_scaled = StandardScaler().fit_transform(df)
pca = PCA()
pca.fit(df_scaled)
df1 = pca.transform(df_scaled)
df1
df2 = pd.DataFrame(df1)
df2
print (pca.explained_variance_ratio_)
import numpy as np
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
per_var

# scree plot

import matplotlib.pyplot as plt
labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]
plt.bar(x = range(1,len(per_var)+1),height = per_var,tick_label = labels)
plt.ylabel("percentage of explained variance")
plt.xlabel('Principal component')
plt.title('Scree Plot')
plt.show()

# accuracy score using the transformed data set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df2,digits.target,test_size=0.2)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
logreg.score(X_test,y_test)


# specifying PCA to cosndier features with 95% variance

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
type(digits)
dir(digits)
#finding the length of each row
l = len(digits.data[0])
#finding the # of rows in the new dataframe
digits.data.shape
#creating the dataframe by appedning rows
import pandas as pd
df = pd.DataFrame(columns = [str(x) for x in range(0,64)])
df

for i in range(1797):
   df.loc[i] = digits.data[i]
df.head(5)
type(df)
df.shape
df_scaled95 = StandardScaler().fit_transform(df)
pca = PCA(0.95)
pca.fit(df_scaled95)
df95 = pca.transform(df_scaled95)
df95
df95_new = pd.DataFrame(df95)
df95_new
print (pca.explained_variance_ratio_)
import numpy as np
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
per_var
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df95_new,digits.target,test_size=0.2)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
logreg.score(X_test,y_test)



