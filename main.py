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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file1 = "simpsons_characters.csv"
file2 = "simpsons_episodes.csv"
file3 = "simpsons_locations.csv"
file4 = "simpsons_script_lines.csv"
characters = pd.read_csv(file1)


def get_data():
    lines = pd.read_csv(file4, low_memory=False, error_bad_lines=False, encoding='utf-8', warn_bad_lines=False)
    lines = lines.iloc[:, :13]
    data = lines[lines.speaking_line == 'true'][['location_id', 'normalized_text', 'character_id']].dropna()
    data['location_id'] = [str(int(i)) for i in data['location_id']]
    return data


def set_targets(data, target_characters):
    global characters
    targets = data['character_id']
    mainChars = [unicode(characters['id'][characters['normalized_name'] == character].values[0])
                 for character in target_characters]
    targets.loc[~targets.isin(mainChars)] = '1000'
    return targets


def train_test_split(data):
    randomselect = np.random.rand(len(data)) < 0.8
    traindata = data[randomselect]
    testdata = data[~randomselect]
    trainY = targets[randomselect]
    testY = targets[~randomselect]
    return traindata, testdata, trainY, testY


def test_log_reg(trainX, trainY, testX, testY):
    model = LogisticRegression()
    model.fit(trainX, trainY)
    prediction = model.predict(testX)
    print 'LogReg', 1 - accuracy_score(testY, prediction)


def test_dec_tree(trainX, trainY, testX, testY):
    for i in range(1, 30, 5):
        clf = DecisionTreeClassifier(max_depth=i)
        clf.fit(trainX, trainY)
        hypothesis = clf.predict(testX)
        print i, 'DecTree', 1 - accuracy_score(testY, hypothesis)


def test_rand_for(trainX, trainY, testX, testY):
    for i in range(1, 30, 5):
        clf = RandomForestClassifier(max_depth=i)
        clf.fit(trainX, trainY)
        hypothesis = clf.predict(testX)
        print i, 'RF', 1 - accuracy_score(testY, hypothesis)


###Takes as CL arguments characters to test for against all other characters
if __name__ == "__main__":
    data = get_data()
    targets = set_targets(data,sys.argv[1:])
    traindata, testdata, trainY, testY = train_test_split(data)
    trainX, train_word_vec, train_loc_vec = extract_features(traindata)
    testX , a, b = extract_features(testdata, train_word_vec, train_loc_vec)
    test_log_reg(trainX, trainY, testX, testY)
    test_dec_tree(trainX, trainY, testX, testY)
    test_rand_for(trainX, trainY, testX, testY)

