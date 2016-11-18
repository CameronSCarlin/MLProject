from sklearn.tree import DecisionTreeClassifier
from StringIO import StringIO
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

# TODO Random Forest: add features and ability to take spread of arguments
# TODO Tree Visualization: add graphical parameters to make trees prettier

#############################
####### Random Forest #######
#############################

# Returns predicted Y classifications to be used in Misclassification test.

def DecisionTreeFn(Xtrain, Xtest, Ytrain, maxDepth):
    DecTree = DecisionTreeClassifier(max_depth=maxDepth)
    DecTree.fit(Xtrain,Ytrain)
    YPredict = DecTree.predict(Xtest)
    return YPredict

##################################
####### Tree Visualization #######
##################################

# Given an X and Y set of data and a max depth of the tree,
# outputs a PDF plot to the given output file.

def TreeToPDF(X, Y, maxDepth, OutputFileName):
    DecTree = DecisionTreeClassifier(max_depth=maxDepth)
    DecTree.fit(X,Y)

    dotfile = StringIO()
    export_graphviz(DecTree, out_file=dotfile)
    graph = graph_from_dot_data(dotfile.getvalue())
    graph.write_pdf('%s.pdf' % OutputFileName)
    return

