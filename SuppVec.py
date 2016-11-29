######################################
####### Support Vector Machine #######
######################################

from sklearn import svm

def SVMfn(Xtrain, Xtest, Ytrain):
    clf = svm.SVC()
    clf.fit(Xtrain, Ytrain)
    YPredict = clf.predict(Xtest)
    return YPredict