# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:53:37 2019

@author: 140524
"""

# Module 3 - case study 1

# question 1

#A Robot moves in a Plane starting from the origin point (0,0). The robot can move toward UP, 
#DOWN, LEFT, RIGHT. The trace of Robot movement is as given following:
#UP 5
#DOWN 3
#LEFT 3
#RIGHT 2
#The numbers after directions are steps. Write a program to compute the distance current position 
#after sequence of movements.
#Hint: Use math module.



direction = 0
steps = 0
command = (direction, steps)
command_list = []
origin = {"x": 0, "y": 0}
while direction is not '':
    direction = input("Direction (U, D, L, R):")
    steps = int(input("Number of steps:"))
    command = (direction, steps)
    command_list.append(command)
print(command_list)

while len(command_list) > 0:
    current = command_list[-1]
    if current[0] == 'U':
        origin["y"] += int(current[1])
    elif current[0] == 'D':
        origin["y"] -= int(current[1])
    elif current[0] == 'L':
        origin["x"] -= int(current[1])
    elif current[0] == 'R':
        origin["x"] += int(current[1])
    command_list.pop()
distance = ((origin["x"])**2 + (origin["y"])**2)**0.5
print(distance)
    
#Weather forecasting organization wants to show is it day or night. So, write 
#a program for such organization to find whether is it dark outside or not.

import time
ltime = time.localtime()
ltime
current_hour = ltime.tm_hour
print(current_hour)

if (current_hour<=7 or current_hour>=19):
    print("Its dark out side")
else:
    print("its day now")    

#Write a program to find distance between two locations when their 
#latitude and longitudes are given.
import math    
coordinates1 = int(input())



#Design a software for bank system. There should be options like cash withdraw,
# cash credit and change password. According to user input, 
#the software should provide required output.    


def withdraw():
    PIN = int(input("enter your PIN: "))
    if PIN == 1234:
        with_amt = int(input("enter the amount: "))
        current_bal = 100000 - with_amt
        print(current_bal)
    else:
        print("your PIN is wrong")
         
def deposit():
    PIN = int(input("enter your PIN: "))
    if PIN == 1234:
        dep_amt = int(input("enter the amount: "))
        current_bal = 100000 + dep_amt
        print(current_bal)
    else:
        print("your PIN is wrong")
        
def change_pswd():         
     PIN = int(input("enter your PIN: "))
     if PIN == 1234:
        x = input("create new pswd: ")
        if x:
            y = input("confirm new pswd: ")
            if x==y:
                 print("your password is set")
            else:
                 print("call cust supprt")
        else:
             pass
     else:
         print("your PIN is wrong")
         
print("enter what you want to do: withdraw, deposit, or change password")         
command = input(" ")
if command == 'withdraw':
               withdraw()
elif command == 'deposit':
               deposit()   
elif command == 'change_password':
           change_pswd()     
else:
    pass             


#----------------
    
#Write a program which will find all such numbers which are divisible by 7 
#but are not a multiple of 5, between 2000 and 3200 (both included).The numbers 
#obtained should be printed in a comma-separated sequence on a single line

listA = [x for x in range(2000,3201)]
listA
listB = list(filter(lambda x:x%7==0 and x%5!=0,listA))
listB
def conv_str(x):
    return str(x)
listC = [conv_str(x) for x in listB]
listC
print(','.join(listC))

#--------

#Write a program which can compute the factorial of a given numbers.
#Suppose the following input is supplied to the program: 8 Then, the output should be:40320

 

def fact(n):
    if n==0:
        return 1
    else:
        return n*fact(n-1)

n=int(input('enter the number: '))
fact(n)
#----------
#Write a program that calculates and prints the value according to the given formula:

#Q = Square root of [(2 * C * D)/H]

#Following are the fixed values of C and H:

#C is 50. H is 30.

#D is the variable whose values should be input to your program in a 
#comma-separated sequence.For example Let us assume the following comma 
#separated input sequence is given to the program:

import math
c=50
h=30
d = input("enter the sequence:").split(',')
result = [str(round(math.sqrt(2*c*int(i)/h))) for i in d]
print(','.join(result))

# or

import math
c=50
h=30
string = input("enter the sequence:")
numlist = string.split(',')
resultlist=[]
for i in numlist:
    temp = round(math.sqrt(2*c*int(i)/h))
    resultlist.append(temp)
resultlist=[str(i) for i in resultlist]    
print(','.join(resultlist))
    
#--  OR---
  
import math
c=50
h=30
string = input("enter the sequence:")
numlist = string.split(',')
resultlist = list(map(lambda x:round(math.sqrt(2*50*int(x)/30)),numlist))
resultlist

#---OR

#  The below process will show how to convert a string list to int list
# The .join mnthod method shows to join only 2 strings
import math
def funct(x):
    return (round(math.sqrt(2*50*x/30)))

string = input("enter the sequence:")
numlist = string.split(',')
intnumlist = [int(x) for x in numlist]
resultlist = [funct(x) for x in intnumlist]
resultlist

#-------

#Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. 
#The element value in the i-th row and j-th column of the array should be i * j.
#Note: i=0,1.., X-1; j=0,1,¡­Y-1. Suppose the following inputs are given to the program: 3,5
#Then, the output of the program should be:
#[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]

dim = input("enter the dim of the array:").split(',')
dimnum = [int(x) for x in dim]
dimnum
result = []
for i in range(dimnum[0]):
    rowlist=[]
    for j in range(dimnum[1]):
         rowlist.append(i*j)
    result.append(rowlist)     
result        


# input splitted ','
dim = input("enter the dim of the array:").split(',')
# convert the input to list and convert the list elements to int
dim = [int(x) for x in dim]
resultlist=[]
for i in range(dim[0]):
    rowlist=[]
    for j in range(dim[1]):
                rowlist.append(i*j)
                #rowlist
    resultlist.append(rowlist)  
print(resultlist)          

#  OR

dim = input("enter the dim of the array:").split(',')
# convert the input to list and convert the list elements to int
dim = [int(x) for x in dim]
resultlist=[[0 for row in range(dim[0])] for col in range(dim[1])]
for row in range(dim[0]):
    for col in range(dim[1]):
        resultlist[row][col] = row*col
              
print(resultlist)    

#----------

#Write a program that accepts a comma separated sequence of words as input and
# prints the words in a comma-separated sequence after sorting them alphabetically.

#Suppose the following input is supplied to the program:

string_in = input("enter the string:").split(',')
print(','.join(sorted(string_in)))


#or


string_in = input("enter the string:").split(',')
string_in
print(sorted(string_in))

#or - to not print a list just a string

string_in = input("enter the string:").split(',')
string_in.sort()
print(','.join(string_in))

#----------

#Write a program that accepts sequence of lines as input and prints the lines 
#after making all characters in the sentence capitalized
#Suppose the following input is supplied to the program:
# cannot do in above method becoz upper functionality is not applible to list object
#----or

listA = []
string_in = input("enter the string:").split(' ')
for element in string_in:
    listA.append(element.upper())
print(','.join(listA))    


#or



listA=[]
string_in = input("enter the string:").split(' ')
for x in string_in:
    list.append(x.upper())
print(' '.join(list))    

#-----or

list = []
while True:
    string_in = input("enter the number:").split(' ')
    if string_in:
          list.append(string_in.upper())
    else:    
          break
for line in list:
    print(line)       

#---------------------
    
#Write a program that accepts a sequence of whitespace separated words as input 
#and prints the words after removing all duplicate words and sorting them alphanumerically.    

string_in = input("enter the string:").split(' ')
result = set(string_in)
print(','.join(sorted(result)))

# or

listU=[]
string_in = input("enter the string:").split(' ')
for i in string_in:
    if i not in listU:
        listU.append(i)
print(','.join(sorted(listU)))        
# or
string_in = input("enter the string:").split(' ')
#sentence = [x for x in string_in.split(' ')]
print(' '.join(sorted(set(string_in))))



#---OR
string_in = input("enter the string:")
sentence = [x for x in string_in.split(' ')]
sentence
print(' '.join(sorted(list(set(sentence)))))
 
#--OR
#this method doesnt work if there are more than 2 repeatations
word = input('enter the string: ').split()

for i in word:
    if word.count(i) > 1:  # count function returns total repeatation of an element that is send as argument
        word.remove(i)     # removes exactly one element per call

word.sort()
print(" ".join(word))


#---OR

word = input("Enter:").split()
[word.remove(i) for i in word if word.count(i) > 1 ]
# removal operation with comprehension method
word.sort()
print(" ".join(word))     
    
    
    
#--------------------    


#Write a program which accepts a sequence of comma separated 4 digit binary numbers 
#as its input and then check whether they are divisible by 5 or not. 
#The numbers that are divisible by 5 are to be printed in a comma separated sequence.

bin_num=input("enter the decimal number:").split(',')
result = []
for i in bin_num:
      y=int(i,2)
      if y%5==0:
          result.append(y)
result          

# or

bin_num=input("enter the decimal number:").split(',')
dec_list = [x for x in bin_num]
dec_list
int_list = []
for x in dec_list:
   y=int(x,2)
   int_list.append(y)
print(int_list)   
result_list = []
for y in int_list:
    if y%5==0:
        result_list.append(y)
    else:
        continue
result_list     

#---OR--best apprach
bin_num=input("enter the decimal number:").split(',')
result_list = []
for x in bin_num:
   y=int(x,2)
   if y%5==0:
      result_list.append(x)
   else:
       continue
print(result_list)  

#---OR

bin_num=input("enter the decimal number:").split(',')
result_list = []
for x in bin_num:
   y=int(x,2)
   if y%5==0:
      result_list.append(x)
   else:
       continue
print(','.join(result_list))  



#-----------
#Write a program that accepts a sentence and calculate the number of upper case 
#letters and lower case letters.Suppose the following input is supplied to the program:

sentence = input("enter the sentence:")
sentence
upper = []
lower = []
for i in sentence:
    if i.isupper():
        upper.append(i)
    elif i.islower():
        lower.append(i)
    else:
        continue
upper
lower
        
#-------------- CASE STUDY 2

cust_in = input("enter the prof and age:",).split(',')
)

import pandas as pd
df = pd.read_csv('C:\\edureka\\PR_Class_3\\Data_set\\bank-data.csv')
df.head()
joblist = list(df.job.unique())
joblist
cust_in = input("enter the prof:",)
if cust_in in joblist:
    print('call cust')
else:
    print("Don't waste calling")        

df.describe().age
df.groupby('y').age.agg(['max','min'])
joblist = list(df.job.unique())
joblist
while True:
    cust_in = input("enter the prof and age:",).split(',')
    job = cust_in[0]
    age = int(cust_in[1])
    if job == 'End':
       print('exiting the program')
       break 
    elif (job in joblist) & (19<=age<=80):
        print('call cust')
    elif (job not in joblist):
        print('job out of bounds')
    else:    
        print('dont call this cust')
    
        

import pandas as pd
df = pd.read_csv('C:\\edureka\\PR_Class_3\\Data_set\\bank-data.csv')
df.head()
print (df.keys())
prof_set = set(df.job)
prof_set
command = input("enter the profession: ")
if command in prof_set:
         print("eligible to call")
else:
         print("Don't waste calling")    


#------------enhancement
age_list = list(df.age)
final_age_list = []
for age in age_list:
    if age not in final_age_list:        
             final_age_list.append(age)
sorted_age = sorted(final_age_list)   
Range_age = [sorted_age[0],sorted_age[len(sorted_age)-1]] 
Range_age

Range_dict = {'Min_age':Range_age[0], 'Max_age':Range_age[1]}  
Range_dict 
    
#---------without OOPs---------using opandas

import pandas as pd
#pd.set_option('display.max_columns', 10)
# reading csv file from url 
df = pd.read_csv('C:\\edureka\\PR_Class_3\\Data_set\\FairDealCustomerData.csv')
df.head()
# adding column headers
df_header = pd.DataFrame(df.values,columns =['Last_Name','Firstname','Decision_Flag'])
df_header.head()
# new data frame with split column - Firstname into Title and First_Name
df_header[['Title','First_Name']] = df_header.Firstname.str.split(".", expand = True)
df_header.head()
# Dropping old Name columns - Firstname
df_header.drop(columns =['Firstname'], inplace = True)
# Rearranging columns sequence
column_list = ['Title','First_Name','Last_Name','Decision_Flag']
df_header = df_header.reindex(columns = column_list)
df_header
# selecting rows based on condition 
df_header_exception = df_header[df_header['Decision_Flag'] == 0] 
df_header_exception        



#----------with OOPs - without Pandas

class employees:
    def __init__(self, title, fname, lname, decisionflag):
       self.__title = title
       self.__firstName = fname
       self.__lastName = lname
       self.__decisionflag = int(decisionflag)
     
#------CASE STUDY 3 
with open('C:\\edureka\\PR_Class_3\\Data_set\\FairDealCustomerData.csv','r') as fread:
  for data in fread:  
     data = data.rstrip('\n')
     title = data.split(',')[1].strip().split('.')[0]
     print(title)
  for 
  
  
  while rec!= [' ']:
            rec = row[1].split(".") + rec[2:]
            employees.append(employees(rec[0], rec[1], rec[2], rec[3])
            rec = fread.readline().strip().split(",")
    

#-----------
            
import re
class CustomerNotAllowedException():
    pass

class Customer():
    title=""
    firstname=""
    lastname=""
    isblacklisted=0
    custList = []
    def __init__(self):
        customerdata=open('C:\edureka\PRClass_3\Data_set\FairDealCustomerData.csv', 'r')
        customerdata
        custList = []
        for data in customerdata:
            data = data.rstrip('\n')
            lastname = data.split(',')[0]
            firstname = data.split('.')[0]
            firstname = firstname.split(',')[0]
            title = data.split(',')[1]
            title = title.split('.')[0]
            isblacklisted = data.split(',')[2]
            self.custList.append(title.strip()+firstname+lastname+" "+isblacklisted)

        customerdata.close()
        # l = [self.custList.index(i) for i in self.custList if 'Mr Allen Allen' in i]
        # print l

    def CreateOrder(self,custName):
        try:
            # custName = raw_input('Enter a customer name:')
            l = [self.custList.index(i) for i in self.custList if self.custName in i]
            print(l)
            # if bool(re.search('1',self.custList[l]))==True:
            #     raise CustomerNotAllowedException

        except CustomerNotAllowedException:
            pass

        else:
            pass

obj=Customer()            
    
        
