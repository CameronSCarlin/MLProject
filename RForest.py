#############################
####### Random Forest #######
#############################

from sklearn.tree import DecisionTreeClassifier

def DecisionTreeFn(Xtrain, Xtest, Ytrain, maxDepth):
    DecTree = DecisionTreeClassifier(max_depth=maxDepth)
    DecTree.fit(Xtrain,Ytrain)
    YPredict = DecTree.predict(Xtest)
    return YPredict

# TODO add features and ability to take spread of arguments