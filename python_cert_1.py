# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:55:10 2019

@author: 140524
"""

#Write a program which will find factors of given number and find whether the factor 
#is even or odd

n = int(input("enter your number: "))
even = []
odd = []
for i in range(2,n):
    if n%i ==0:
        if i%2==0:
            even.append(i)
        else:
            odd.append(i)
    else:
        continue
even
odd

#Write a code which accepts a sequence of words as input and prints the words in 
#a sequence after sorting them alphabetically.

In = input("enter the string: ").split(',')
print(In)
print(sorted(In))

#Write a program, whichwill find all the numbers between 1000 and 3000 (both included) 
#such that each digit of a number is an even number. The numbers obtained should be printed 
#in a comma separated sequence on a single line
result = []
for i in range(1000,3000):
    temp = str(i)
    if (int(temp[0])%2==0) and (int(temp[1])%2==0) and (int(temp[2])%2==0) and (int(temp[3])%2==0):
        result.append(temp)
result        

#Write a program that accepts a sentence and calculate the number of letters and digits.

In = input("enter the string: ")
str_digit=[]
str_alpha=[]
str_digit=[x for x in In if x.isdigit()]
str_digit
str_alpha=[x for x in In if x.isalpha()]
str_alpha

#Design a code which will find the given number is Palindrome number or not.Hint:

In = input("enter the number: ")
if In == In[::-1]:
     print('yes')
else:
     print('no')     