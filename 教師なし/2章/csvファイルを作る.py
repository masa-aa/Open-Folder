"""
a.csvを作りx(pd.DataFrame)をa.csvに書き込む
"""
import pandas as pd
x = pd.DataFrame([1,2,3])
import csv
file = open('b.csv','w')
file.close() 
x.to_csv('b.csv')