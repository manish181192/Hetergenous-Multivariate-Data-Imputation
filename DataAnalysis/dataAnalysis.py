import pandas as pd
# import json
datapath = "/home/manish/ML/Rutgers/Project/ML3/ML3/ML3AllSites.csv"

df = pd.read_csv(datapath)
print(len(list(df)))
df = df[['Site', 'age', 'RowNumber']]
# def frequency_table(x):
#     return pd.crosstab(index=x,  columns="count")
# freqDistFile = open("freqDist.txt", "w")
# ctabs = {}
# for column in df:
#     ctabs[column]=frequency_table(df[column])
#     freqDistFile.write(column+"\n")
#     freqDistFile.write(ctabs[column].to_string()+"\n\n\n")
# freqDistFile.close()
