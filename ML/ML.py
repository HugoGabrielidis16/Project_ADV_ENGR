from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prepare_json import *

import json
import numpy

with open("/Users/hugo/Cincinnati/ADV_ENGR/my_env/Project_ADV_ENGR/json/all.json", "r") as read_file:
    data = json.load(read_file)



Review_label = []

for i in range(len(data)):
    a = data[i]["comment"]
    b = data[i]["label"]
    Review_label.append([a,b])


df = pd.DataFrame(Review_label)
df.columns = ["review","label"]


for i in range(len(df)):
    if df["label"][i] == "Bug":
        df[i] = 1
    elif df["label"][i] == "Feature":
        df[i] = 2
    elif df["label"][i] == "Rating":
        df[i] = 3
    elif df["label"][i] == "UserExperience":
        df[i] = 4

X_train, X_test, Y_train, Y_test = train_test_split(df['review'],
                                                    df['label'],
                                                    test_size=0.2,
                                                    stratify=df['label'])

print('Size of Training Data ', X_train.shape[0])
print('Size of Test Data ', X_test.shape[0])





# fit model no training data
""" 
clf_gxb = XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, num_classes=4)
clf_gxb.fit(X_train, Y_train)  

Y_pred = clf_gxb.apply(X_test)
print(print(Y_pred)) """