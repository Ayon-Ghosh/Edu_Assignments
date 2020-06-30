# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:21:32 2019

@author: 140524
"""

#practise edureka_class 11
# Apriori algo

import pandas as pd
df = pd.read_excel('C:\\edureka\\PR_Class_11\\codes_in_class\\Online_Retail.xlsx')
df.head()
df.columns
df.isnull().sum()
df['Description'] = df['Description'].str.strip()
# some of the invoiceNo starts with C, so we have to convert the invoice# column to str
# type
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df[df.InvoiceNo.str.contains('C')]
# filtering out the rows in which invoicenO starts with C 
df = df[~df.InvoiceNo.str.contains('C')]
df.head()

# creating the basket 

basket = df[df.Country=='France'].groupby(['InvoiceNo','Description']).Quantity.sum().unstack().fillna(0)
basket

# we dont care of the quantity of each item. We just need to find out which are 0 and the ones which r not
# so we can mark all that 0 as 0 and anything>0 as 1

def encode(x):
    if x <=0:
        return 0
    if x>=1:
        return 1
basket = basket.applymap(encode)
basket   
basket.drop('postage', axis=1, inplace=True) 


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(basket, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules[ (rules['lift'] >= 5) & (rules['confidence'] >= 0.6) ]