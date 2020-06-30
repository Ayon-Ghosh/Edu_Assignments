# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:19:45 2019

@author: 140524
"""

#Module 2
#Case Study 1
#What is the output of the following code?
#nums =set([1,1,2,3,3,3,4,4])print(len(nums))

#Answer: The len is 3

#What will be the output?
d ={"john":40, "peter":45}
list(d.keys())
print(list(d.keys()))

#A website requires a user to input username and password to register. 
#Write a program to check the validity of password given by user. Following are 
#the criteria for checking password:1. At least 1 letter between [a-z]2. At least 
#1 number between [0-9]1. At least 1 letter between [A-Z]3. At least 1 character
# from [$#@]4. Minimum length of transaction password: 65. Maximum length of transaction 
#password: 12


password = input("enter the password:")
import re
pattern = "^.*(?=.{8,})(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=]).*$"
result = re.findall(pattern, password)
if (result):
    print("valid password:",password.split(','))
else:
    print ("Password not valid" )    
    
    
#Write a for loop that prints all elements of a list and their position in the list

a = [4,7,3,2,5,9] 
i = []
for x in range(len(a)+1):
     i.append(x)
print(dict(zip(i,a)))     

#Please   write   a   program   which accepts  a   string   from   console   and   
#print   the characters that have even indexes

str_in = input("enter the string:")
str_out=[]
for i in range(len(str_in)+1):
    if i%2==0:
        str_out.append(str_in[i])
    else:
        continue
print(''.join(str_out))    
    

#Please write a program which accepts a string from console and print it in reverse order.

str_in = input("enter the string:")
str_out = str_in[::-1]
print(str_out)

#Please write a program which count and print the numbers of each character in a 
#string input by console.

str_in = input("enter the string:")
set_in = set(str_in)
count = []
for i in set_in:
    temp = str_in.count(i)
    count.append(temp)
print(dict(zip(set_in, count))) 

#With   two   given   lists   [1,3,6,78,35,55]   and   [12,24,35,24,88,120,155],   
#write   a program to make a list whose elements are intersection of the above given lists.

listA =  [1,3,6,78,35,55]
setA = set(listA)
listB =  [12,24,35,24,88,120,155] 
setB = set(listB)
setC = setA.intersection(setB)
list(setC)

#With a given list [12,24,35,24,88,120,155,88,120,155], write a program to print 
#this list after removing all duplicate values with original order reserved

listA = [12,24,35,24,88,120,155,88,120,155]
setA = set(listA)
print(list(setA))

# or

listA = [12,24,35,24,88,120,155,88,120,155]
listB = []
for i in listA:
    if i not in listB:
        listB.append(i)
listB        

#By using list comprehension, please write a program to print the list after 
#removing the value 24 in [12,24,35,24,88,120,155].
listA = [12,24,35,24,88,120,155]
listB = [x for x in listA if x!=24]
listB


#By using list comprehension, please write a program to print the list after 
#removing the 0th,4th,5th numbers in [12,24,35,70,88,120,155].

listA = [12,24,35,24,88,120,155]
listB = [x for (i,x) in enumerate(listA) if i!=0 and i!=4 and i!=5]
listB

#By using list comprehension, please write a program to print the list after 
#removing delete numbers which are divisible by 5 and 7 in [12,24,35,70,88,120,155]

listA = [12,24,35,24,88,120,155]
listB = [x for x in listA if x%35!=0]
listB

#Please  write  a  program  to  randomly  generate  a  list  with  5  numbers, 
# which  are divisible by 5 and 7 , between 1 and 1000 inclusive

import random
print(random.sample([i for i in range(1,1001) if i%35==0], 5))

#Write  a  program  to  compute  1/2+2/3+3/4+...+n/n+1  with  a  given  n  input  
#by console (n>0).

n = int(input('enter the number:'))
result = 0
for i in range(1,n+1):
    result = result +i/(i+1)
print(round(result,2))    
