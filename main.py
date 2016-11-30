###############################
###### Final Executable #######
###############################

import Validations, LogReg, KNNClass, RForest, SuppVec, Features
import pandas as pd
from Features import extract_features, top_words
import numpy as np
from sklearn.externals import joblib
from collections import Counter
import sys


file1 = "simpsons_characters.csv"
file2 = "simpsons_episodes.csv"
file3 = "simpsons_locations.csv"
file4 = "simpsons_script_lines.csv"


characters = pd.read_csv(file1)
episodes = pd.read_csv(file2)
locations = pd.read_csv(file3)
lines = pd.read_csv(file4, low_memory=False, error_bad_lines=False, encoding='utf-8', warn_bad_lines=False)
lines = lines.iloc[ : , :13]


#subset of data we need
data = lines[lines.speaking_line=='true'][['location_id', 'normalized_text','character_id']].dropna()
#convert location_id from float to integer to string
data['location_id'] = [str(int(i)) for i in data['location_id']]

targets = data['character_id']


#mainChars = list(zip(*Counter(targets.tolist()).most_common(4))[0])

names = [sys.argv[i] for i in range(1,len(sys.argv))]
mainChars = [unicode(characters['id'][characters['normalized_name']==name].values[0]) for name in names]

targets.loc[~targets.isin(mainChars)] = '1000'

### Method to construct list of top words for feature construction is called below
### saved word list in words.csv and am loading to save time while testing

#words = np.array(top_words(data))
words = np.loadtxt('words.csv', delimiter=',', dtype='S')


#WILL RUN OUT OF MEMORY if ran on whole set, try on subset
#we should explore sparse matrices?

randomselect = np.random.rand(len(data)) < 0.8
traindata = data[randomselect]
testdata = data[~randomselect]

trainX, train_word_vec, train_loc_vec = extract_features(traindata)
testX , a, b = extract_features(testdata, train_word_vec, train_loc_vec)
trainY = targets[randomselect]
testY = targets[~randomselect]
# select features
#features = Features.feature_selection(features,targets, 10)




#simply test for Logisticregression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


model = LogisticRegression()
model.fit(trainX, trainY)
prediction = model.predict(testX)
print 'LogReg', 1 - accuracy_score(testY, prediction)

for i in range(1,30,5):
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(trainX, trainY)
    hypothesis = clf.predict(testX)
    print i, 'DecTree', 1 - accuracy_score(testY, hypothesis)

for i in range(1,30,5):
    clf = RandomForestClassifier(max_depth=i)
    clf.fit(trainX, trainY)
    hypothesis = clf.predict(testX)
    print i, 'RF', 1 - accuracy_score(testY, hypothesis)

