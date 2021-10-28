from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prepare_json import *


seed = 7
test_size = 0.33
print(DataFrame_json_review_Matrix.shape)
print(DataFrame_json_label_Matrix.shape)
X_train, X_test, y_train, y_test = train_test_split(DataFrame_json_review_Matrix, DataFrame_json_label_Matrix, test_size=test_size,
 random_state=seed,stratify = DataFrame_json_label_Matrix )



# fit model no training data

clf_gxb = XGBClassifier(objective = "binary:logistic",missing = None, seed = seed)
clf_gxb.fit(X_train,y_train,verbose=True,early_stopping_rounds=10,eval_metric="aucpr",
eval_set=[(X_test,y_test)])

print(clf_gxb)