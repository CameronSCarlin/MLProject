###################################
####### Logistic Regression #######
###################################

from sklearn.linear_model import LogisticRegression

def LogReg(*arg):
    if len(arg) < 3:
        print "Needs trainX, testX, trainY at a minimum"
        return
    if len(arg) == 3:
        model = LogisticRegression()
        model.fit(arg[0],arg[2])
        prediction = model.predict(arg[1])
        return prediction
    if len(arg) == 4:
        model = LogisticRegression()
        model.fit(arg[0],arg[2])
        prediction = model.predict(arg[1])
        return prediction
    if len(arg) > 4:
        print "Too many arguments for current implementation"
        return
