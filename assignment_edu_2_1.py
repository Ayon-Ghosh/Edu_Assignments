# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:47:24 2019

@author: 140524
"""


# separate digits and print in a list

num = int(input("enter your num: "))
i = num
n = len(str(i))
ListA = list()
while(num!=0):
   div = num%10
   ListA.append(div)
   num = num//10
print(ListA)
print(sorted(ListA))

#  Write a program that accepts a sentence and calculate 
#the number of letters and digits.

Main = input("enter the string: ")
listA=[x for x in Main if x.isalpha()]
print(listA, "and the length is:",len(listA))
listN=[x for x in Main if x.isdigit()]
print(listN, "and the length is:",  len(listN))

#Write a program which will find factors of given number and 
#find whether the factor is even or odd

num=int(input("enter your number: "))
ListO = list()
ListE = list()
for i in range(2,num):
    if num%i==0:
        if i%2==0:
            ListE.append(i)
        else:
             ListO.append(i)
print(ListE)
print(ListO)      


#Write a program, whichwill find all the numbers between 1000 and 3000 
#(both included) such that each digit of a number is an even number. 
#The numbers obtained should be printed in a comma separated sequence on 
#a single line 

listEdigs = []
for i in range(1000,3001):    
    numstr = str(i)
    if(int(numstr[0])%2==0) and (int(numstr[1])%2==0) and (int(numstr[2])%2==0) and (int(numstr[3])%2==0):
        listEdigs.append(numstr)
print(listEdigs)


#--Write a code which accepts a sequence of words as input and prints the words
# in a sequence after sorting them alphabetically.

str1 = input("enter your words: ")
print(sorted(str1.split()))


#Design a code which will find the given number is Palindrome number or not.

str1 = input("enter your string: ")
str2 = str1[::-1]
if str1==str2:
    print("Bingo: your string is pallindrom")
else:
    print("sorry, out of luck")    
    
    
#--output - questions from edureka casestudy 2

nums = set([1,1,2,3,3,3,4,4]) 
print(len(nums))  

d = {"john":40, "peter":45} 
print(list(d.keys()))  

d = {"john":40, "peter":45} 
print(d.keys())


# A website requires a user to input username and password to register. 
#Write a program to check the validity of password given by user. 
#Following are the criteria for checking password:
#At least 1 letter between [a-z]
#2. At least 1 number between [0-9]
#1. At least 1 letter between [A-Z]
#3. At least 1 character from [$#@]
#4. Minimum length of transaction password: 6
#5. Maximum length of transaction password: 12


import re
pattern = "^.*(?=.{8,})(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=]).*$"
password = input("Enter string to test: ")
result = re.findall(pattern, password)
if (result):
    print("valid password:",password.split(','))
else:
    print ("Password not valid" )     


#Write a for loop that prints all elements of a list and their position 
#in the list.

a = [4,7,3,2,5,9]
for i in range(len(a)):
    print(a[i], "=>", a.index(a[i]))
    
#Please write a program which accepts a string from console and print the characters that have even indexes.
#Example: If the following string is given as input to the program:
#H1e2l3l4o5w6o7r8l9d
#Then, the output of the program should be:
#Helloworld

str1 = input("enter your string: ")
result = []
for i in range(len(str1)):
          if i%2==0:
             result.append(str1[i])
print(''.join(result))        
        
    
#Please write a program which accepts a string from console and 
#print it in reverse order.    
    
str1 = input("enter your string: ")
str2 = str1[::-1]    
print(str2)

#Please write a program which count and print the numbers of each character in
# a string input by console.

str1 = input("enter your string: ")
for i in range(len(str1)):
    print(str1[i], "=>", str1.count(str1[i]))
    
#With two given lists [1,3,6,78,35,55] and [12,24,35,24,88,120,155], write a 
#program to make a list whose elements are intersection of the above given lists. 
    
list1=[1,3,6,78,35,55]
a=set()
for i in range(len(list1)):
    a.add(list1[i])
a  

list2=[12,24,35,24,88,120,155]    
b=set()
for i in range(len(list2)):
    b.add(list2[i])
b
set(a&b)  

#By using list comprehension, please write a program to print the list after 
#removing the value 24 in [12,24,35,24,88,120,155].

[x for x in [12,24,35,24,88,120,155] if x!=24]

#By using list comprehension, please write a program to print the list after 
#removing the 0th,4th,5th numbers in [12,24,35,70,88,120,155].

list1 = [12,24,35,24,88,120,155]
[x for x in list1 if list1.index(x)!=0 and list1.index(x)!=4 and list1.index(x)!=5]


#By using list comprehension, please write a program to print the list after 
#removing delete numbers which are divisible by 5 and 7 in [12,24,35,70,88,120,155].

list1 = [12,24,35,24,88,120,155]
[x for x in list1 if x%35!=0]

#Write a program to compute 1/2+2/3+3/4+...+n/n+1 with a given n input by 
#console (n>0).

num=int(input("enter your number: "))
result = float(0)

for i in range(1,num+1):
    result = float(result)+float(i/(i+1))

print(result)    

