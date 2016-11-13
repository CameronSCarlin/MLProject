import csv
import sys

def readcsv(filename):
    templist = []
    with open(filename) as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            templist.append(row)
    return headers, templist

headers, TrainData = readcsv('train.csv')
headers, TestData = readcsv('test.csv')

print headers
print TrainData[0]
print TestData[0]