import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

file1 = "simpsons_characters.csv"
file4 = "simpsons_script_lines.csv"
characters = pd.read_csv(file1)

#retrieves required data and formats correctly
def get_data():
    lines = pd.read_csv(file4, low_memory=False, error_bad_lines=False, encoding='utf-8', warn_bad_lines=False)
    lines = lines.iloc[:, :13]
    data = lines[lines.speaking_line == 'true'][['location_id', 'normalized_text', 'character_id']].dropna()
    data['location_id'] = [str(int(i)) for i in data['location_id']]
    return data

#given dataset and list of characters to test against all other, returns targets
def set_targets(data, target_characters):
    global characters
    targets = data['character_id']
    mainChars = [unicode(characters['id'][characters['normalized_name'] == character].values[0])
                 for character in target_characters]
    targets.loc[~targets.isin(mainChars)] = '1000'
    return targets

#splits data into train/test
def train_test_split(data, targets):
    randomselect = np.random.rand(len(data)) < 0.8
    traindata = data[randomselect]
    testdata = data[~randomselect]
    trainY = targets[randomselect]
    testY = targets[~randomselect]
    return traindata, testdata, list(trainY), list(testY)

#last list in list of lists is 'everyone else'
def CorpusMaker(data,list):
    numlist = len(list) + 1
    corpi = [[] for i in xrange(numlist)]
    mainChars = [unicode(characters['id'][characters['normalized_name'] == character].values[0])
                 for character in list]
    for row in data.itertuples():
        if row[3] == '1000':
            corpi[numlist-1].append(row[2])
        else:
            location = mainChars.index(row[3])
            corpi[location].append(row[2])
    return corpi

#returns most common words overall for all groups
def Joiner(corpi, numwords):
    JoinedCorpiSet = set()
    for i in range(len(corpi)):
        joinedstring = ' '.join(corpi[i])
        stringlist = joinedstring.split()
        counter = Counter(stringlist).most_common(numwords)
        wordset = set(zip(*counter)[0])
        JoinedCorpiSet = JoinedCorpiSet | wordset
    return list(JoinedCorpiSet)

def listAndLength(TrainOrTest):
    fractionslist = []
    for row in TrainOrTest.itertuples():
        words = row[2].split()
        tempCounter = Counter(words)
        templength = float(len(words))
        linelist = []
        for word in overallSet:
            linelist.append(tempCounter[word] / templength)
        linelist.append(templength)  # adds in length of list
        linelist.append(sum(map(len, words)) / float(len(words)))  # adds in average word length
        linelist.append(len(max(words, key=len)))  # adds in length of longest term
        fractionslist.append(linelist)
    return fractionslist

def confusion(testY, hypothesis):
    confuse = confusion_matrix(testY, hypothesis)
    tpr = float(confuse[1, 1]) / (float(confuse[1, 0]) + float(confuse[1, 1]))
    fpr = float(confuse[0, 1]) / (float(confuse[0, 0]) + float(confuse[0, 1]))
    print confuse
    print "true positive %f" % tpr
    print "false positive %f" % fpr

def RF(trainx, trainy, testx, testy, listtoiter):
    for i in listtoiter:
        clf = RandomForestClassifier(max_depth=i)
        clf.fit(trainx, trainy)
        hypothesis = clf.predict(testx)
        if len(Iwant) == 1:
            confusion(testY, hypothesis)
        print "RF Misclass (%d) = %f" % (i, (1 - accuracy_score(testy, hypothesis)))

def Dectree(trainx, trainy, testx, testy, listtoiter):
    for i in listtoiter:
        clf = DecisionTreeClassifier(max_depth=i)
        clf.fit(trainx, trainy)
        hypothesis = clf.predict(testx)
        if len(Iwant) == 1:
            confusion(testY, hypothesis)
        print "Dectree Misclass (%d) = %f" % (i, (1 - accuracy_score(testy, hypothesis)))

def LR(trainx, trainy, testx, testy):
    clf = LogisticRegression(n_jobs=-1, penalty='l1')
    clf.fit(trainx, trainy)
    hypothesis = clf.predict(testx)
    if len(Iwant) == 1:
        confusion(testY, hypothesis)
    print "LogReg Misclass L1 = %f" % (1 - accuracy_score(testy, hypothesis))
    clf = LogisticRegression(n_jobs=-1, penalty='l2')
    clf.fit(trainx, trainy)
    hypothesis = clf.predict(testx)
    if len(Iwant) == 1:
        confusion(testY, hypothesis)
    print "LogReg Misclass L2 = %f" % (1 - accuracy_score(testy, hypothesis))

def Ada(trainx, trainy, testx, testy):
    clf = AdaBoostClassifier()
    clf.fit(trainx, trainy)
    hypothesis = clf.predict(testx)
    if len(Iwant) == 1:
        confusion(testY, hypothesis)
    print "Ada Misclass = %f" % (1 - accuracy_score(testy, hypothesis))

def bag(trainx, trainy, testx, testy):
    clf = BaggingClassifier()
    clf.fit(trainx, trainy)
    hypothesis = clf.predict(testx)
    if len(Iwant) == 1:
        confusion(testY, hypothesis)
    print "Bagging Misclass = %f" % (1 - accuracy_score(testy, hypothesis))

def SupVec(trainx, trainy, testx, testy):
    clf = SVC()
    clf.fit(trainx, trainy)
    hypothesis = clf.predict(testx)
    if len(Iwant) == 1:
        confusion(testY, hypothesis)
    print "SupVec Misclass = %f" % (1 - accuracy_score(testy, hypothesis))

#top ten most common characters in order
Iwant = ['homer simpson','marge simpson', 'bart simpson', 'lisa simpson',
         'c montgomery burns', 'moe szyslak', 'seymour skinner', 'ned flanders',
         'grampa simpson', 'milhouse van houten'] ####### character list is here, needs at least 1

if __name__ == "__main__":
    data = get_data()
    targets = set_targets(data,Iwant)
    treedepths = [25,50,75]     ####### RF and DecTree loop through this list of depths
    traindata, testdata, trainY, testY = train_test_split(data, targets)
    corpi = CorpusMaker(traindata, Iwant)
    numberWord = 250            ####### Change # to number of words desired per character
    overallSet = Joiner(corpi, numberWord)
    print "This execution is using %d characters and %d words per character" % (len(Iwant), numberWord)
    traindata = listAndLength(traindata)
    testdata = listAndLength(testdata)
    RF(traindata, trainY, testdata, testY, treedepths)      ####### Comment out an algorithm if you don't want it to execute
    Dectree(traindata, trainY, testdata, testY, treedepths)
    LR(traindata, trainY, testdata, testY)
    Ada(traindata, trainY, testdata, testY)
    bag(traindata, trainY, testdata, testY)
    SupVec(traindata, trainY, testdata, testY)




