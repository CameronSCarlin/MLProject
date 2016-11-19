#######################################
####### K-Fold Cross Validation #######
#######################################




######################################
####### Misclassification Rate #######
######################################


def misClassification(prediction, testY, testX):
    totalMisclassification = []
    totalMisclassification.append(float(sum(prediction != testY)) / len(testX))
    print totalMisclassification
