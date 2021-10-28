
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import pandas as pd
from prepare_json import *
from function import *




# split into train/test sets

seed = 7
test_size = 0.1
trainX, testX, trainy, testy =  train_test_split(DataFrame_json_review_Matrix, DataFrame_json_label_Matrix,test_size=test_size )


from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB(alpha=1)

#affichage(trainX)
#affichage(trainy)
clf.fit(trainX, trainy)
 
""" print(clf.predict(testX))
print(testy) """


def accuracy(prediction,ground_truth):
    sum = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            sum +=1
    return sum/len(prediction)



#affichage(clf.predict_proba(testX))
print(accuracy(clf.predict(testX),testy))