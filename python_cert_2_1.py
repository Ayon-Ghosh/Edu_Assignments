# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:43:48 2019

@author: 140524
"""
#Module 2
#case study 2
str_in = input('enter the reference_ID:')
import re
import base64
pattern = "^.*(?=.{8,})(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=]).*$"
result = re.findall(pattern, str_in)
if (result):
    str2 = base64.b64encode(str_in.encode('utf-8'))
    print(str2)
else:
    print ("Reference_ID not valid" )    
print(base64.b64decode(str2))    
