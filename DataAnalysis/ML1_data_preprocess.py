import csv

path = '/home/manish/ML/Rutgers/Project/ML1/ML1/Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv'

file=open( path, "r")
reader = csv.reader(file)
for line in reader:
    print(line)