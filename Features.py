#############################################
####### Data Reader / Feature Creator #######
#############################################

import csv

def readcsv(filename):
    templist = []
    with open(filename) as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            templist.append(row)
    return headers, templist

headers, ScriptData = readcsv('simpsons_script_lines.csv')
print headers
print ScriptData[0]

#TODO
# remove the ones where speaking_line (6th column) == False
# replace character_id with the actual character names from the character csv?
# same with location
# use character as Y, use location, normalized text, and word count to create features (also maybe episode id?)
# manually create corpus of person specific terms or phrases?

# TODO Features to add
# TFIDF?
# dummy variables of location
# stemmed words vs actual words?