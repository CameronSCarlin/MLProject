######################################
####### Support Vector Machine #######
######################################

## Nothing in here is tested, just a warning...

from sklearn import svm

def SVMfn(Xtrain, Xtest, Ytrain):
    clf = svm.SVC()
    clf.fit(Xtrain, Ytrain)

## TODO not finished...