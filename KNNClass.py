##################################
####### KNN Classification #######
##################################

from sklearn.neighbors import KNeighborsClassifier

# TODO add more arguments if necessary

# Returns predicted classsifications
# First 4 arguments ALWAYS are Xtrain, Xtest, Ytrain, NumNeighbors
# Argument #5 is weights, can be 'uniform' or 'distance'

def KNNfn(*arg):
    if len(arg) < 4:
        print "Needs Xtrain, Xtest, Ytrain, NumNeighbors at a minimum"
        return
    if len(arg) == 4:
        neigh = KNeighborsClassifier(n_neighbors=arg[3])
        neigh.fit(arg[0],arg[2])
        YPredict = neigh.predict(arg[1])
        return YPredict
    if len(arg) == 5:
        neigh = KNeighborsClassifier(n_neighbors=arg[3], weights = arg[4])
        neigh.fit(arg[0],arg[2])
        YPredict = neigh.predict(arg[1])
        return YPredict
    if len(arg) > 5:
        print "Too many arguments for current implementation"
        return