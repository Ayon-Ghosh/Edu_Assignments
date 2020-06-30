# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:11:05 2019

@author: 140524
"""

#Module 6 
#case study 1

#Load the dataset “prisoners.csv” using pandas and display the first and last five rows in the dataset.
import pandas as pd
df=pd.read_csv('C:\edureka\PRClass_6\Data_set\prisoners.csv')
df.head(5)
df.tail(5)

#printing first 5 and last 5 rows at the same time
import pandas as pd
df1=pd.read_csv('C:\edureka\PRClass_6\Data_set\prisoners.csv',header=None)
df1
df1.shape
df1.head(5)
df1.tail(5)
for i in range(36):
    if (i>5) & (i<31):
         df1.drop(i,axis=0,inplace=True)
df1         

#Use describe method in pandas and find out the number of columns. 
#Can you say something about those rows who have zero inmates?         
import pandas as pd
my_df=pd.read_csv('C:\edureka\PRClass_6\Data_set\prisoners.csv')
my_df.loc[(my_df['No. of Inmates benefitted by Elementary Education']==0)|
        (my_df['No. of Inmates benefitted by Adult Education']==0)|
        (my_df['No. of Inmates benefitted by Higher Education']==0)]

#Data Manipulation:
#Create a new column -’total_benefitted’ that is a sum of inmates benefitted through all modes.
import pandas as pd
my_df=pd.read_csv('C:\edureka\PRClass_6\Data_set\prisoners.csv')
my_df.dtypes
my_df['Total_Ben'] = my_df['No. of Inmates benefitted by Elementary Education']+my_df['No. of Inmates benefitted by Adult Education']+my_df['No. of Inmates benefitted by Higher Education']+my_df['No. of Inmates benefitted by Computer Course']
my_df['Total_Ben']
my_df.head(5)

#Plotting:
#Make a bar plot with each state name on the x -axis and their total benefitted 
#inmates as their bar heights. Which state has the maximum number of beneficiaries?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_df=pd.read_csv('C:\edureka\PRClass_6\Data_set\prisoners.csv')
my_df.dtypes
my_df['Total_Ben'] = my_df['No. of Inmates benefitted by Elementary Education']+my_df['No. of Inmates benefitted by Adult Education']+my_df['No. of Inmates benefitted by Higher Education']+my_df['No. of Inmates benefitted by Computer Course']
my_df['Total_Ben']
fig,ax = plt.subplots()  
labels = np.array(my_df['STATE/UT'])
y=np.array(my_df['Total_Ben'])
ind = np.arange(len(labels))
#plt.xticks(labels)
plt.xlabel('states')
plt.ylabel('Total_Ben')
width = 0.5
ax.bar(ind,y, width)
ax.set_xticks(ind+width/2)
ax.set_xticklabels(labels, rotation=90)
plt.show()

#Make a pie chart that depicts the ratio among different modes of benefits.
import matplotlib.pyplot as plt
plt.figure(figsize=(3,3))
my_df.columns
x=[my_df['No. of Inmates benefitted by Elementary Education'].sum(),my_df['No. of Inmates benefitted by Adult Education'].sum(),my_df['No. of Inmates benefitted by Higher Education'].sum(),my_df['No. of Inmates benefitted by Computer Course'].sum()]
x
labels = ['No. of Inmates benefitted by Elementary Education','No. of Inmates benefitted by Adult Education','No. of Inmates benefitted by Higher Education','No. of Inmates benefitted by Computer Course']
plt.pie(x,labels = labels)
plt.show()

#Case Study 2

#Load the data from “cereal.csv” and plot histograms of sugar and vitamin 
#content across different cereals.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.xlabel('x-axis')
plt.ylabel('y-axis')
df=pd.read_csv('C:\edureka\PRClass_6\Data_set\cereal.csv')
plt.hist([df['sugars'],df['vitamins']])


#The names of the manufactures are coded using alphabets, 
#create a new column with their full name using the below mapping.
#'N': 'Nabisco',
#'Q': 'Quaker Oats',
#'K': 'Kelloggs',
#'R': 'Raslston Purina',
#'G': 'General Mills' ,
#'P' :'Post' ,
#'A':'American Home Foods Products'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cereal=pd.read_csv('C:\edureka\PRClass_6\Data_set\cereal.csv')
Manu=[]
dict = {'N': 'Nabisco',
'Q': 'Quaker Oats',
'K': 'Kelloggs',
'R': 'Raslston Purina',
'G': 'General Mills' ,
'P' :'Post' ,
'A':'American Home Foods Products'}
cerealM = cereal['mfr']
type(cerealM)
for i in cerealM:
    if i =='N':
        Manu.append(dict['N'])
    elif i=='Q':    
         Manu.append(dict['Q'])
    elif i=='K':    
         Manu.append(dict['K'])
    elif i=='R':    
         Manu.append(dict['R'])   
    elif i=='G':    
         Manu.append(dict['G'])  
    elif i=='P':    
         Manu.append(dict['P'])  
    elif i=='A':    
         Manu.append(dict['A'])   
    else:
         Manu.append('Nan')   
Manu         
cereal['Manufacturer'] = pd.Series(Manu)
cereal.head(5)
         
# or-- shorter
M=[]
cerealM = cereal['mfr']
for i in cerealM:
    for key in dict:
          if i==key:
             M.append(dict[key]) 
          
    
M       
cereal['Manu'] = pd.Series(M)
cereal.head(5)

#Create a bar plot where each manufacturer is on the y axis and the height of 
#the bars depict the number of cereals manufactured by them.


#drink.groupby('continent').beer_servings.agg(['count','min','max','mean'])
select_df = cereal.groupby('Manu').Manu.agg(['count'])
select_df.reset_index(inplace=True)
select_df
plt.style.use('ggplot')
y = list(np.array(select_df['count']))
y
x = list(np.array(select_df['Manu']))
x
fig,ax = plt.subplots()    
width = 0.75
ind = np.arange(len(y)) 
for i, v in enumerate(y):
    print(i,v)
    ax.text(v+1, i - .25, str(v), color='orange', fontweight='bold')
ax.barh(ind,y,width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Manufacturer Prods')
plt.xlabel('No. of prod')
plt.ylabel('Manu')      
plt.show()

#Extract the rating as your target variable ‘y’ and all numerical parameters as 
#your predictors ‘x’. Separate 25% of your data as test set.

#Example of Multivariate linear regresssion 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model,metrics
cereal=pd.read_csv('C:\edureka\PRClass_6\Data_set\cereal.csv')
cereal.dtypes
ListC=['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups']
X = cereal[ListC]
Y= cereal['rating']
reg=linear_model.LinearRegression()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
reg.fit(X_train,Y_train)
reg.coef_
reg.intercept_
X_train

#Fit a linear regression module and measure the mean squared error on test dataset.

y_pred=reg.predict(X_test)
y_pred
df=pd.DataFrame({'Actual data':Y_test,'Predicted Data':y_pred})
df  
print('Mean squared error:',metrics.mean_squared_error(Y_test,y_pred))


# case study 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('C:\edureka\PRClass_6\Data_set\FyntraCustomerData.csv')
data.dtypes

#Compute -- Use seaborn to create a jointplot to compare the Time on Website and 
#Yearly Amount Spent columns. Is there a correlation?
sns.jointplot(x='Time_on_Website',y='Yearly_Amount_Spent',data=data,kind = 'reg')


#Compute – Do the same as above but now with Time on App and Yearly Amount Spent. 
#Is this correlation stronger than 1st One?
sns.jointplot(x='Time_on_App',y='Yearly_Amount_Spent',data=data,kind = 'reg')
# yes - stronger correlation

#Compute -- Explore types of relationships across the entire data set using pairplot . 
#Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
feature_cols = ['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership']
X = data[feature_cols]
Y = data['Yearly_Amount_Spent']
sns.pairplot(data,x_vars=['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership'], y_vars = 'Yearly_Amount_Spent',size=5, aspect=0.7,kind='reg')
#Length of membership is most strongly correlated

#Compute – Create linear model plot of Length of Membership and Yearly Amount Spent. 
#Does the data fits well in linear plot?
X = data['Length_of_Membership']
Y = data['Yearly_Amount_Spent']
plt.plot(X,Y)
# No the date doesnt fit into a linear plt

#Compute – Train and Test the data and answer multiple questions
feature_cols = ['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership']
X = data[feature_cols]
Y = data['Yearly_Amount_Spent']
from sklearn import datasets, linear_model,metrics
reg=linear_model.LinearRegression()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
reg.fit(X_train,Y_train)
#Compute – Predict the data and do a scatter plot. Check if actual and 
#predicted data match?
y_pred = reg.predict(X_test)
y_pred

df=pd.DataFrame({'Actual data':Y_test,'Predicted Data':y_pred})
df  
sns.pairplot(df,x_vars=['Actual data'], y_vars = 'Predicted Data',size=5, aspect=0.7,kind='reg')

# or

plt.scatter(df['Actual data'],df['Predicted Data'])

#What is the value of Root Mean Squared Error?
print('Root Mean squared error:',np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))

