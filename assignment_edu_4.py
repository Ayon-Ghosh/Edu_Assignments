# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 10:24:30 2019

@author: 140524
"""

#Module 4 -Numpy, Pandas, and Matplotlib
# Case Study 1

#Extract data from the given SalaryGender CSV file and store the data from each 
#column in a separate NumPy array
import pandas as pd
import numpy as np
table = pd.read_csv("C:\edureka\PR_class_4\Data_Set\SalaryGender.csv")
table
table.Salary
np.array(table.Salary)
#  OR
sal = np.array(table.loc[:,['Salary']])
print(sal)

table.Gender
np.array(table.Gender)

# OR
Gen = np.array(table.loc[:,['Gender']])
print(Gen)

table.PhD
np.array(table.PhD)

#  OR
degree = np.array(table.loc[:,['PhD']])
print(degree)

#--OR
import pandas as pd
import numpy as np
table = np.genfromtxt("C:\edureka\PR_class_4\Data_Set\SalaryGender.csv",delimiter=',')
sal = np.array(table[:,0])
print(sal)
Gen = np.array(table[:,1])
print(Gen)
Age = np.array(table[:,2])
print(Age)

#Find: 1. The number of men with a PhD 
#2. The number of women with a PhD
import pandas as pd
import numpy as np
table = pd.read_csv("C:\edureka\PR_class_4\Data_Set\SalaryGender.csv")
table[(table.PhD==1) & (table.Gender==1)].count()

# OR----
selected_data = table.loc[:,['Gender','PhD']]

selected_data       
count=0
for index,row in selected_data.iterrows():
     Male = row['Gender']
     Phd = row['PhD']
     if Male==1 and Phd==1:
               print(row)
               count=count+1
print('Male and Phd:', count)  


 #~~~~~~~~~~~~~~~~~~
 
import pandas as pd
import numpy as np
table = pd.read_csv("C:\edureka\PR_class_4\Data_Set\SalaryGender.csv")
table[(table.PhD==1) & (table.Gender==0)].count()

#---OR
import pandas as pd
import numpy as np
table = pd.read_csv("C:\edureka\PR_class_4\Data_Set\SalaryGender.csv")
selected_data = table.loc[:,['Gender','PhD']]

selected_data       
count=0
for index,row in selected_data.iterrows():
     Female = row['Gender']
     Phd = row['PhD']
     if Female==0 and Phd==1:
               print(row)
               count=count+1
print('Female and Phd:', count)              

#Use SalaryGender CSV file. Store the “Age” and “PhD” columns in one DataFrame 
#and delete the data of all people who don’t have a PhD
#creating a smaller table with specific rows
import pandas as pd
import numpy as np
table = pd.read_csv("C:\edureka\PR_class_4\Data_Set\SalaryGender.csv",usecols=['Age','PhD'])
table.shape
table[table.PhD==0]
table[table.PhD==0].shape

# OR----

new_table = table.loc[:,['Age','PhD']]
new_table
# converting table to dataframe
df = pd.DataFrame(new_table,columns = ['Age','PhD'])
# filtering a data frame
df_filtered = df[df['PhD'] == 1]   
df_filtered       

#Calculate the total number of people who have a PhD degree from SalaryGender CSV file.

selected_data       
count=0
for index,row in selected_data.iterrows():
     Phd = row['PhD']
     if Phd==1:
               print(row)
               count=count+1
print('Phd:', count)  

# or

print(len(df_filtered))

# OR-----

table[table.PhD==1].count()

#How do you Count The Number Of Times Each Value Appears In An Array Of Integers?
#[0, 5, 4, 0, 4, 4, 3, 0, 0, 5, 2, 1, 1, 9]
#Answer should be array([4, 2, 1, 1, 3, 2, 0, 0, 0, 1]) which means 0 comes 4 times, 
#1 comes 2 times, 2 comes 1 time, 3 comes 1 time and so on.  

#[0, 5, 4, 3, 2, 1, 9]
listA = [0, 5, 4, 0, 4, 4, 3, 0, 0, 5, 2, 1, 1, 9]
listB=[]
listC=[]
for i in listA:
     if i not in listB:
         listB.append(i)
  
for i in listB:
    temp = listA.count(i)
    listC.append(temp)
listB
print('-------')    
listC    
# to print in dict
my_dict={}
for i in range(len(listB)):
    my_dict[listB[i]] = listC[i]
my_dict    
         
# OR----using groupby

a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
from itertools import groupby
for key,group in groupby(a):
        print(key,'|',a.count(key))
        
# OR----using groupby        
        
for key, group in groupby(a): 
    sub_list = list(group)
    print(len(sub_list),'|', sub_list)        
# using list comprehension
listb = [len(list(group)) for key,group in groupby(a)]   
listb         

#Create a numpy array [[0, 1, 2], [ 3, 4, 5], [ 6, 7, 8],[ 9, 10, 11]]) and 
#filter the elements greater than 5.

import numpy as np
a = np.array([[0, 1, 2], [ 3, 4, 5], [ 6, 7, 8],[ 9, 10, 11]])
a[a>=5]

# OR-------
import numpy as np
a = np.array([[0, 1, 2], [ 3, 4, 5], [ 6, 7, 8],[ 9, 10, 11]])
print(a)
np.array(a[:2,:])


# OR
import numpy as np
a = np.array([[0, 1, 2], [ 3, 4, 5], [ 6, 7, 8],[ 9, 10, 11]])
print(a)
filter_a = np.squeeze(np.array(a[:2,:]))
#filter_a = np.array(filter(lambda x:x<5,a))
print(filter_a)

# or

a = np.arange(6).reshape(2,3)
print(a)

# or creating an array whose elements after 5 is 0

a = np.array([[0, 1, 2], [ 3, 4, 5], [ 6, 7, 8],[ 9, 10, 11]])
#m = a>5


a[a>5]=a[a>5]*100
a
# or
a = np.array([[0, 1, 2], [ 3, 4, 5], [ 6, 7, 8],[ 9, 10, 11]])
m=a>5
a[m] = a[m]*0
a[m]
a

#Create a numpy array having NaN (Not a Number) and print it.
#array([ nan, 1., 2., nan, 3., 4., 5.])
#Print the same array omitting all elements which are nan
import numpy as np
a = np.array([ 'NaN', 1., 2., 'NaN', 3., 4., 5.])
a[a!='NaN']

# OR
import numpy as np
a = np.array([ 'NaN', 1., 2., 'NaN', 3., 4., 5.])
np.array([x for x in a if x!='NaN'])


#Create a 10x10 array with random values and find the minimum and maximum
#values.
import numpy as np
x = np.random.randn(10,10)
xmin = x.min()
xmax = x.max()
print(xmin, xmax)

#Create a random vector of size 30 and find the mean value.

import numpy as np
Z = np.random.randn(30)
Z
print(Z.mean())

#Create numpy array having elements 0 to 10 And negate all the elements between
#3 and 9
arr=np.arange(11)
arr
arr[(arr>3)&(arr<9)]=arr[(arr>3)&(arr<9)]*-1
arr
# creating a boolean index and storing it in m
# refer: https://www.python-course.eu/numpy_masking.php
m = (arr > 3) & (arr < 9)
#m prints only boolean values(redundant step - good for learning)
m
# converting m to int values 0 and 1(redundant step - good for learning)
print(m.astype(int))
#arr[m] prints that part of the array only
arr[m]=arr[m]* -1
arr

#Create a random array of 3 rows and 3 columns and sort it according to 1st column,
#2nd column or 3rd column.
import numpy as np
import pandas as pd
a = pd.DataFrame(np.random.randn(3,3),columns=['col1','col2','col3'])
a
a.sort_values('col1')
a.sort_values('col2')
a.sort_values('col3')

#Create a random array and swap two rows of an array.

import pandas as pd
drink = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')  
drink_sublist = drink.head(5)
drink_sublist
temp=drink_sublist.loc[0]
drink_sublist.loc[0]=drink_sublist.loc[1]
drink_sublist.loc[1]=temp
drink_sublist

# last question - Tenesse school
#Phase1
import pandas as pd
school = pd.read_csv('C:\edureka\PR_class_4\Data_Set/middle_tn_schools.csv') 
school.head() 
school.columns
school.dtypes
school.describe()
#phase2

school.groupby('school_rating').reduced_lunch.mean()


#phase 3
import matplotlib.pyplot as plt
plt.xlabel('school_rating')
plt.ylabel('reduced_lunch')
school.groupby('school_rating').reduced_lunch.mean().plot()

#phase 4
 
import matplotlib.pyplot as plt
import scipy as sci
x = np.array(school.reduced_lunch)
y = np.array(school.school_rating)
plt.ylabel('school_rating')
plt.xlabel('reduced_lunch')
p1 = sci.polyfit(x,y,1)
print(p1)
plt.scatter(x,y)
plt.plot(x.polyval(p1,x),'r-')

# phase 5 - remaining




# Case Study 2

import pandas as pd

#--maths
math = pd.read_csv('C:\edureka\PR_class_4\Data_Set\MathScoreTerm1.csv') 
math.head()
math.drop(['Name','Ethinicity','Subject','Age','Sex'],axis=1,inplace=True)
math.columns
math_cols = ['Math_Score','ID']
math.columns = math_cols
math.columns
math.head()
#--physics

physics = pd.read_csv('C:\edureka\PR_class_4\Data_Set\PhysicsScoreTerm1.csv') 
physics.head()
physics.drop(['Name','Ethinicity','Subject'],axis=1,inplace=True)
physics.columns
physics_cols = ['Physics_Score','Age','Sex','ID']
physics.columns = physics_cols
physics.columns
physics.head()

merge1 = pd.merge(math,physics, on='ID',how='left')
merge1.head()
# DS
DS = pd.read_csv('C:\edureka\PR_class_4\Data_Set\DSScoreTerm1.csv') 
DS.head()
DS.drop(['Name','Ethinicity','Subject','Age','Sex'],axis=1,inplace=True)
DS.columns
DS_cols = ['DS_Score','ID']
DS.columns = DS_cols
DS.columns
DS.head()

final_merge = pd.merge(merge1,DS, on='ID',how='left')
final_merge.head()
final_merge.loc[0,'Sex']
booleans = []
for i in final_merge.Sex:
#    print(final_merge.iloc[i,'Sex'])
    
    if i=='M':
        booleans.append(True)
    else:
        booleans.append(False)
booleans
is_long = pd.Series(booleans)
final_merge['Sex'] = is_long.astype(int)
final_merge


# OR----

final_merge['Sex']=final_merge.Sex.str.contains('M').astype(int)
final_merge.head()

# filling missing values

final_merge['Math_Score'].fillna(value=0)
final_merge['Physics_Score'].fillna(value=0)
final_merge['DS_Score'].fillna(value=0)
final_merge['Sex']=final_merge.Sex.str.contains('M').astype(int)
final_merge.head()
final_merge.to_csv('C:\python\ScoreFinal.csv')

#----Enhancement
import pandas as pd
ScoreFinal = pd.read_csv('C:\python\ScoreFinal.csv')
ScoreFinal['DS_Score'].fillna(0,inplace =True)
ScoreFinal['Physics_Score'].fillna(0,inplace =True)
ScoreFinal['Math_Score'].fillna(0,inplace =True)
ScoreFinal
DS = pd.read_csv('C:\edureka\PR_class_4\Data_Set\DSScoreTerm1.csv') 
DS.head()
E = DS.Ethinicity
final_merge_Eth = pd.concat([ScoreFinal,E],axis=1)
final_merge_Eth.Ethinicity[final_merge_Eth.Ethinicity=='White American']=100
final_merge_Eth.Ethinicity[final_merge_Eth.Ethinicity=='European American']=200
final_merge_Eth.Ethinicity[final_merge_Eth.Ethinicity=='White American']=300
final_merge_Eth.Ethinicity[final_merge_Eth.Ethinicity=='Hispanic']=300
final_merge_Eth
x=round(final_merge_Eth['DS_Score'].mean())
final_merge_Eth.DS_Score[final_merge_Eth.DS_Score==0]=x
y=round(final_merge_Eth['Physics_Score'].mean()) 
final_merge_Eth.Physics_Score[final_merge_Eth.Physics_Score==0]=y
z=y=round(final_merge_Eth['Math_Score'].mean()) 
final_merge_Eth.Math_Score[final_merge_Eth.Math_Score==0]=z
final_merge_Eth.head(60)

# Case Study 3

#You are given a dataset, which is present in the LMS, containing the number of
#hurricanes occurring in the United States along the coast of the Atlantic. Load the
#data from the dataset into your program and plot a Bar Graph of the data, taking
#the Year as the x-axis and the number of hurricanes occurring as the Y-axis.

import pandas as pd
import numpy as np
H = pd.read_csv('C:\edureka\PR_class_4\Data_Set\Hurricanes.csv')
import matplotlib.pyplot as plt
plt.xlabel('Year')
plt.ylabel('Hurricanes')
x = np.array(H.Year)
y = np.array(H.Hurricanes)
plt.bar(x,y)

#The dataset given, records data of city temperatures over the years’ 2014 and
#2015. Plot the histogram of the temperatures over this period for the cities of
#San Francisco and Moscow.
import pandas as pd
df = pd.read_csv('C:\edureka\PR_class_4\Data_Set\CityTemps.csv')
plt.xlabel('City Temp')
plt.ylabel('Year')
plt.hist([df["San Francisco"],df["Moscow"]])
plt.show()


#Create csv file from the data file available in LMS which goes by the name
#‘M4_assign_dataset’ and read this file into a pandas data frame
import pandas as pd
import numpy as np
a = pd.read_csv('C:\edureka\PR_class_4\Data_file\data_file.txt',delimiter=',')
a.head()
a.to_csv('C:\python\Ayon_Test.csv')

#Let the x axis data points and y axis data points are
#X = [1,2,3,4]
#y = [20, 21, 20.5, 20.8]
#5.1: Draw a Simple plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
X = [1,2,3,4]
Y = [20, 21, 20.5, 20.8]
plt.plot(X,Y)

#Configure the line and markers in simple plot
# configure line
fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
ax2 = fig1.add_subplot(222)
ax3 = fig1.add_subplot(223)
ax1.plot(X,Y,'-',color='green',linewidth=3)
ax2.plot(X,Y,'--',color='red',linewidth=1.5)
ax3.plot(X,Y,'-.',color='blue',linewidth=5)


# or ---

fig_2,axes=plt.subplots(1,3,figsize=(15,8))
axes[0].plot(X,Y,'-',color='green',linewidth=3)
axes[1].plot(X,Y,'--',color='red',linewidth=1.5)
axes[3].plot(X,Y,'-.',color='blue',linewidth=5)
plt.show()
# configure marker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
X = [1,2,3,4]
Y = [20, 21, 20.5, 20.8]
fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
ax2 = fig1.add_subplot(222)
ax3 = fig1.add_subplot(223)
ax1.plot(X,Y,linestyle = (0,(5,1)),marker = '+',markersize = 10, color='green')
ax2.plot(X,Y,linestyle = (0,(3,5,1,5)),marker = '.',markersize = 12,color='red')
ax3.plot(X,Y,linestyle = (0,(1,1)),marker = 'x',markersize = 8,color='blue')
#configure the axes
#Give title of Graph & labels of x axis and y axis
plt.xlim(2,4)
plt.ylim(20,21)
plt.plot(X,Y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Learning Matplotlib')

#Give error bar if y_error = [0.12, 0.13, 0.2, 0.1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
X = [1,2,3,4]
Y = [20, 21, 20.5, 20.8]
y_error = [0.12, 0.13, 0.2, 0.1]
plt.errorbar(X,Y,yerr=y_error,fmt='o')
plt.show()

#Draw a scatter graph of any 50 random values of x and y axis

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(50)
y = np.random.randn(50)
plt.scatter(x,y)
plt.show()

#Create a dataframe from following data
#'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
#'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
#'female': [0, 1, 1, 0, 1],
#'age': [42, 52, 36, 24, 73],
#'preTestScore': [4, 24, 31, 2, 3],
#'postTestScore': [25, 94, 57, 62, 70]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],'gender': [0, 1, 1, 0, 1],'age': [42, 52, 36, 24, 73],'preTestScore': [4, 24, 31, 2, 3],'postTestScore': [25, 94, 57, 62, 70]})
df
df_new = df.sort_values('age')

#Draw a Scatterplot of preTestScore and postTestScore, with the size of each point
#determined by age

x = np.array(df_new.preTestScore)
y = np.array(df_new.postTestScore)
age = np.array(df_new.age)
s = [10*2**n for n in range(len(age))]
plt.scatter(x,y,s=s)
plt.show()

#Draw a Scatterplot from the data in question 9 of preTestScore and
#postTestScore with the size = 300 and the color determined by sex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],'gender': [0, 1, 1, 0, 1],'age': [42, 52, 36, 24, 73],'preTestScore': [4, 24, 31, 2, 3],'postTestScore': [25, 94, 57, 62, 70]})
df
df.plot.scatter('preTestScore', 'postTestScore', c='gender', s = 300, colormap='jet')
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~Case Study 4~~~~~~~~~~~~~~~~~~~~

#Plot Total Sales Per Month for Year 2011. How the total sales have increased over 
#months in Year 2011. Which month has lowest Sales?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sales = pd.read_csv('C:\edureka\PR_class_4\Data_Set\BigMartSalesData.csv')
sales.head()
sales.columns
sales.shape
sales.dtypes
sales11=sales[sales.Year==2011]
sales11.shape
salesM=sales11.groupby('Month').Amount.agg(['sum','min','max','mean'])
salesM
salesM[salesM['sum']==salesM['sum'].min()]
salesM.reset_index(inplace=True)
salesM
#Plot Total Sales Per Month for Year 2011 as Bar Chart. Is Bar Chart Better to 
#visualize than Simple Plot?
x= np.array(salesM.Month)
x
y= np.array(salesM['sum'])
y
plt.xlabel('Month')
plt.ylabel('Monthly_Sales')
plt.bar(x,y)

#Plot Pie Chart for Year 2011 Country Wise. Which Country contributes highest 
#towards sales?
# also pulling out the largest pie and adding %s to pie

salesC=sales11.groupby('Country').Amount.agg(['sum','min','max','mean'])
salesC.reset_index(inplace=True)
x = salesC['sum']
labels = salesC['Country']
labels
plt.figure(figsize=(10,10))
plt.pie(x,labels = labels)
plt.show()

#Plot Scatter Plot for the invoice amounts and see the concentration of amount. 
#In which range most of the invoice amounts are concentrated
salesIA=sales11.groupby('InvoiceNo').Amount.agg(['sum'])
salesIA.reset_index(inplace=True)
salesIA
plt.scatter(salesIA['InvoiceNo'],salesIA['sum'])
plt.show()


#Enhancements for code
#Change the bar chart to show the value of bar
#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
S= np.array(salesM['sum'])
x = list(map(lambda x:round(int(x)/10000), S))
x
y= np.array(salesM.Month)
y
M=['JAN','FEB','MAR','ARL','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
ind = np.arange(len(M))
ind
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
width = 0.5
for key,value in enumerate(x):
    ax.text(value-2,key+0.25,str(value),color='m', fontweight='bold')
ax.set_yticks(ind+width/2)
ax.set_yticklabels(M,minor=False)    
#make some space between bars by align and also adjusting width
ax.bar(x,ind,width,color='b',align='edge',label = 'Monthly Sales Total')
plt.xlabel('Sales_Total')
plt.ylabel('Month')
plt.legend()
plt.show()

# same things HZ bar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
S= np.array(salesM['sum'])
x = list(map(lambda x:round(int(x)/10000), S))
x
y= np.array(salesM.Month)
y
M=['JAN','FEB','MAR','ARL','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
ind = np.arange(len(M))
ind
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
width = 0.5
for key,value in enumerate(x):
    ax.text(value+2,key+0.25,str(value),color='m', fontweight='bold')
ax.set_yticks(ind+width/2)
ax.set_yticklabels(M,minor=False)    
#make some space between bars by align and also adjusting width
ax.barh(ind,x, width,color='b',align='edge',label = 'Monthly Sales Total')
plt.xlabel('Sales_Total')
plt.ylabel('Month')
plt.legend()
plt.show()

#In Pie Chart Play With Parameters shadow=True, startangle=90 and see how different the chart looks
salesC=sales11.groupby('Country').Amount.agg(['sum','min','max','mean'])
salesC.reset_index(inplace=True)
x = salesC['sum']
labels = salesC['Country']
labels
plt.figure(figsize=(10,10))
plt.pie(x,labels = labels,shadow=True,startangle=90)
plt.show()

#In scatter plot change the color of Scatter Points

salesIA=sales11.groupby('InvoiceNo').Amount.agg(['sum'])
salesIA.reset_index(inplace=True)
salesIA
plt.scatter(salesIA['InvoiceNo'],salesIA['sum'],c='gender', s = 5, colormap='jet')
plt.show()


#---------------------END--------------