# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:33:44 2019

@author: 140524
"""

#Module 5--
#Case Study 1

#From the data provided on Hollywood movies:
#Find the highest rated movie in the “Quest” story type.
import pandas as pd
movie = pd.read_csv('C:\edureka\PR_class_5\Data_set\HollywoodMovies.csv')
movie_sel = movie[movie.Story=='Quest']
movie_sort = movie_sel.sort_values('RottenTomatoes',ascending=False)
movie_sort.reset_index(inplace=True)
movie_sort.head(5)
movie_sort.loc[0,['Movie']]

#Find the genre in which there has been the greatest number of movie releases
import pandas as pd
movie = pd.read_csv('C:\edureka\PR_class_5\Data_set\HollywoodMovies.csv')
movie.groupby('Genre').Genre.count().sort_values(ascending=False)

#Print the names of the top five movies with the costliest budgets.
import pandas as pd
movie = pd.read_csv('C:\edureka\PR_class_5\Data_set\HollywoodMovies.csv')
movie.head()
movie.loc[:,['Movie','Budget']].sort_values('Budget',ascending=False).head(5)

#Is there any correspondence between the critics’ evaluation of a movie and its 
#acceptance by the public? Find out, by plotting the net profitability of a movie
# against the ratings it receives on Rotten Tomatoes.
import pandas as pd
import matplotlib.pyplot as plt
movie = pd.read_csv('C:\edureka\PR_class_5\Data_set\HollywoodMovies.csv')
plt.xlabel('Rating')
plt.ylabel('Profit')
plt.scatter(movie["RottenTomatoes"],movie["Profitability"])
plt.show()

#Perform Operations on Files
#From the raw data below create a data frame
#Save the dataframe into a csv file as example.csv
#'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
#'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
#'female': [0, 1, 1, 0, 1],
#'age': [42, 52, 36, 24, 73],
#'preTestScore': [4, 24, 31, 2, 3],
#'postTestScore': [25, 94, 57, 62, 70]

import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
df = pd.DataFrame({'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],'gender': [0, 1, 1, 0, 1],'age': [42, 52, 36, 24, 73],'preTestScore': [4, 24, 31, 2, 3],'postTestScore': [25, 94, 57, 62, 70]})
df.to_csv('C:\python\example.csv.csv')
#Read the example.csv without column heading
df_new = pd.read_csv('C:\python\example.csv.csv',header=None)
df_new
df_new.drop(0,axis=1).loc[1:,:]

#Read the example.csv and make the index columns as 'First Name’ and 'Last Name'
df_new = pd.read_csv('C:\python\example.csv.csv')
df_new
df_new.drop('Unnamed: 0',axis=1,inplace=True)
df_new['name'] = df_new.first_name+' '+df_new.last_name
df_new.set_index('name')

#Print the data frame in a Boolean form as True or False. True for Null/ NaN values and false for non-null values
#Read the dataframe by skipping first 3 rows and print the data frame
df_new.loc[3:,:]

#Load a csv file while interpreting "," in strings around numbers as thousands seperators. 
#Check the raw data 'postTestScore' column has, as thousands separator.
#Comma should be ignored while reading the data. 
#It is default behaviour, but you need to give argument to read_csv function 
#which makes sure commas are ignored.

#Perform Operations on Files
#From the raw data below create a Pandas Series 'Amit', 'Bob', 'Kate', 'A', 'b', 'np.nan', 'Car', 'dog', 'cat'
import numpy as np
import pandas as pd
arr=np.array(['Amit', 'Bob', 'Kate', 'A', 'b','np.nan', 'Car', 'dog', 'cat'])
series=pd.Series(arr)
type(series)

#Print all elements in lower case
df=pd.DataFrame(arr)
df
df.loc[:,0].str.lower()
#Print all the elements in upper case
df.loc[:,0].str.upper()

#Print the length of all the elements
listA=[]
for i in df.loc[:,0]:
    x = len(i)
    listA.append(x)
listA    

#From the raw data below create a Pandas Series
#' Atul', 'John ', ' jack ', 'Sam'
import numpy as np
import pandas as pd
pd.Series([' Atul', 'John ', ' jack ', 'Sam'])
#Print all elements after stripping spaces from the left and right
a = pd.Series([' Atul', 'John ', ' jack ', 'Sam'])
listA = []
for i in a:
    x = i.strip(' ')
    listA.append(x)
listA    
b=pd.Series([listA])
b

#Print all the elements after removing spaces from the left only
import numpy as np
import pandas as pd
a=pd.Series([' Atul', 'John ', ' jack ', 'Sam'])
listA = []
for i in a:
    x = i.lstrip(' ')
    listA.append(x)
listA    
b=pd.Series([listA])
b

#Print all the elements after removing spaces from the right only
import numpy as np
import pandas as pd
a=pd.Series([' Atul', 'John ', ' jack ', 'Sam'])
listA = []
for i in a:
    x = i.rstrip(' ')
    listA.append(x)
listA    
b=pd.Series([listA])
b

#Create a series from the raw data below
#'India_is_big', 'Population_is_huge', np.nan, 'Has_diverse_culture'
#split the individual strings wherever ‘_’ comes and create a list out of it.

a=['India_is_big', 'Population_is_huge', 'np.nan', 'Has_diverse_culture']
listA=[]
for i in range(len(a)):
        x =a[i].split('_')  
        listA.append(x)
listA        

#Access the individual element of a list
#Expand the elements so that all individual elements get splitted by ‘_’ and 
#insted of list returns individual elements

for i in range(len(listA)):
    for j in range(len(listA[i])):
               print(listA[i][j])
               
#Create a series and replace either X or dog with XX-XX
#'A', 'B', 'C', 'AabX', 'BacX','', np.nan, 'CABA', 'dog', 'cat'  
import pandas as pd
A = ['A', 'B', 'C', 'AabX', 'BacX','', 'np.nan', 'CABA', 'dog', 'cat']   
series = pd.Series(A)   
listA=[]
for i in series:
    if 'X' in i:
        x=i.replace('X','XX-XX')
        listA.append(x)
    elif 'dog' in i:
        y= i.replace('dog','XX-XX')
        listA.append(y)
    else:
        listA.append(i)
listA
strA = ','.join(listA)   
strA  

#Create a series and remove dollar from the numeric values
#'12', '-$10', '$10,000'

A = ['12', '-$10', '$10,000'] 
B=[]    
for i in A:
    if '$' in i:
        x = i.replace('$','')
        B.append(x)
    else:
        B.append(i)    
B        

#or
A='12', '-$10', '$10,000'
series=pd.Series(A)
series.str.replace(r'$', '')
    
#Create a series and reverse all lower case words
#'india 1998', 'big country', np.nan
A = 'india 1998', 'big country', 'np.nan'
series=pd.Series(A)
X=r'[a-z]+'
repl=lambda m: m.group(0)[::-1]
s=pd.Series(['india 1998', 'big country', np.nan]).str.replace(X,repl)
print(s)

#Create pandas series and print true if value is alphanumeric in series or false 
#if value is not alpha numeric in series.        
#'1', '2', '1a', '2b', '2003c'
import pandas as pd
A='1', '2', '1a', '2b', '2003c'
s=pd.Series(['1', '2', '1a', '2b', '2003c'])
result=[]
isresult = False
for i in s:
    if i.isdigit():
        result.append(isresult)
    elif i.isalpha():
        result.append(isresult)
    else:
        isresult=True
        result.append(isresult)
result        
pd.Series(result)
        
#or

s=pd.Series([1, 2, '1a', '2b', '2003c'])
print(s.str.isalnum())

#Create pandas series and print true if value is containing ‘A’
#'1', '2', '1a', '2b', 'America', 'VietnAm','vietnam', '2003c'
import pandas as pd
s = pd.Series(['1' '2', '1a', '2b', 'America', 'VietnAm','vietnam', '2003c'])
print(s.str.contains('A'))

#Create pandas series and print in three columns value 0 or 1 is a or b or c exists in values

#Create pandas dataframe having keys and ltable and rtable as below -
#'key': ['One', 'Two'], 'ltable': [1, 2] 
#'key': ['One', 'Two'], 'rtable': [4, 5]
#Merge both the tables based of key
import pandas as pd
df1 = pd.DataFrame({'key': ['One', 'Two'], 'ltable': [1, 2]})
df1
df2 = pd.DataFrame({'key': ['One', 'Two'], 'rtable': [4, 5]})
df2
print(pd.merge(df1,df2,on='key'))        

#----case study 2

#Compute how much total salary cost has increased from year 2011 to 2014

import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
#type casting int like this doesnt work for large numbers casted as flt64
#sal['TotalPay'] = sal['TotalPay'].astype(int)
#type casting int like below will change
sal['TotalPay'] = sal['TotalPay'].astype(np.int64)
sal.head(5)
sal.dtypes
sal_1 = sal.groupby('Year').TotalPay.agg(['sum'])
sal_1
sal_1.reset_index(inplace=True)
sal_1
import matplotlib.pyplot as plt
plt.xticks([2011,2012,2013,2014])
plt.plot(sal_1['Year'],sal_1['sum'])

#Which Job Title in Year 2014 has highest mean salary?
import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
sal['TotalPay'] = sal['TotalPay'].astype(np.int64)
sal.dtypes
sal_2 = sal[sal.Year==2014].groupby('JobTitle').TotalPay.agg(['mean'])
sal_2['mean'] = sal_2['mean'].astype(np.int64)
sal_2.sort_values('mean',ascending=False)

#How much money could have been saved in Year 2014 by stopping OverTimePay?
import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
sal['OvertimePay'] = sal['OvertimePay'].astype(np.int64)
sal.dtypes
sal[sal.Year==2014].OvertimePay.sum()

#Which are the top 5 common job in Year 2014 and how much do they cost SFO ?
import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
sal['TotalPay'] = sal['TotalPay'].astype(np.int64)
sal[sal.Year==2014].groupby('JobTitle').JobTitle.agg(['count']).sort_values('count',ascending=False).head(5)
x=sal[(sal.Year==2014) & (sal.JobTitle=='Transit Operator')].TotalPay.sum()
y=sal[(sal.Year==2014) & (sal.JobTitle=='Special Nurse')].TotalPay.sum()
z=sal[(sal.Year==2014) & (sal.JobTitle=='Registered Nurse')].TotalPay.sum()
w=sal[(sal.Year==2014) & (sal.JobTitle=='Public Svc Aide-Public Works')].TotalPay.sum()
v=w=sal[(sal.Year==2014) & (sal.JobTitle=='Firefighter')].TotalPay.sum()
a=x+y+z+w+v
print(a)
#enhancements
#Which are the last 5 common job in Year 2014 and how much do they cost SFO?
sal[sal.Year==2014].groupby('JobTitle').JobTitle.agg(['count']).sort_values('count',ascending=False).tail(5)
#same process as above to find the sum

#Who was the top earning employee across all the years?
import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
sal['TotalPay'] = sal['TotalPay'].astype(np.int64)
sal.columns
sal.sort_values('TotalPay',ascending=False).loc[0,'EmployeeName']


##enhancements
#In year 2011 OverTimePay was what percentage of TotalPayBenefits
import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
sal['TotalPay'] = sal['TotalPay'].astype(np.int64)
sal['OvertimePay'] = sal['OvertimePay'].astype(np.int64)
sal.dtypes
x=sal[sal.Year==2011].OvertimePay.sum()
y=sal[sal.Year==2011].TotalPay.sum()
perc = round(x*100/y)
print(perc)

#Which Job Title in Year 2014 has lowest mean salary?
import pandas as pd
import numpy as np
sal = pd.read_csv('C:\edureka\PR_class_5\Data_set\Salaries.csv')
sal['TotalPay'] = sal['TotalPay'].astype(np.int64)
sal[sal.Year==2014].groupby('JobTitle').TotalPay.agg(['mean']).sort_values('mean',ascending=False).tail(5)



