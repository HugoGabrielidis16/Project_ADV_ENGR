
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from prepare_json import *
from function import *

k = 0
current = 500000

x = 3490
y = 3500

X = DataFrame_json_review_Matrix[x:y]

for i in range(1,3670):

    knn = KNeighborsClassifier(n_neighbors=i)



    knn.fit(DataFrame_json_review_Matrix_test ,DataFrame_json_label_Matrix_test )



    prediction = knn.predict(X)

    """ print("Prediction : ")

    print(prediction)

    print("True label : ")

    print(DataFrame_json_label_Matrix[x:y]) """

    if difference(prediction,DataFrame_json_label_Matrix[x:y]) < current:
        k = i
        current = difference(prediction,DataFrame_json_label_Matrix[x:y])
        print(current)
        Matrix = prediction
    
print(k)

print(prediction)

print(DataFrame_json_label_Matrix[x:y])

#print("Accuracy of model at K=4 is",metrics.accuracy_score(y_test, Pred_y))