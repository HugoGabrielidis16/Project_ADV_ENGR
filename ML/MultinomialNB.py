import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import json

from sklearn.feature_extraction.text import TfidfVectorizer
with open("/Users/hugo/Cincinnati/ADV_ENGR/my_env/Project_ADV_ENGR/json/all.json", "r") as read_file:
    data = json.load(read_file)


import nltk
from nltk.corpus import stopwords


Review_label = []

for i in range(len(data)):
    a = data[i]["comment"]
    b = data[i]["label"]
    Review_label.append([a,b])


df = pd.DataFrame(Review_label)
df.columns = ["review","label"]


for i in range(len(df)):
    if df["label"][i] == "Bug":
        df["label"][i] = 1
    elif df["label"][i] == "Feature":
        df["label"][i] = 2
    elif df["label"][i] == "Rating":
        df["label"][i] = 3
    elif df["label"][i] == "UserExperience":
        df["label"][i] = 4



X_train, X_test, Y_train, Y_test = train_test_split(df['review'],
                                                    df['label'].array,
                                                    test_size=0.2,
                                                    stratify=df['label'])

print('Size of Training Data ', X_train.shape[0])
print('Size of Test Data ', X_test.shape[0])

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

countv = CountVectorizer(min_df = 1, ngram_range=(1,5), stop_words="english")
X_train_tf = countv.fit_transform(X_train)

model1 = MultinomialNB()
model1.fit(X_train_tf, Y_train)

X_test_tf = countv.transform(X_test)
Y_pred_NB = model1.predict(X_test_tf)
print ('Accuracy Score MultinomialNB  mind_df = 1, ngram_range=(1,1) : - ', accuracy_score(Y_test, Y_pred_NB)*100)


plot_confusion_matrix(model1,X_test_tf,
                      Y_test, values_format='d',
                      cmap=plt.cm.Blues)
plt.title("Confusion matrix for MultinomialNB, mind_df = 1, ngram_range=(1,1)")
plt.show()

df_t = pd.read_csv("/Users/hugo/Cincinnati/ADV_ENGR/my_env/Project_ADV_ENGR/Database/reviews.csv",sep =";")
X_testing = df["review"]

X_l =  countv.transform(X_testing)

Y_l = model1.predict(X_l)


Bug = []
Feature = []
Rating = []
UserExperience = []
for i in range(len(X_test)):
    if Y_l[i] == 1:
        Bug.append(X_testing.iloc[i])
    elif Y_l[i] == 2:
        Feature.append(X_testing.iloc[i])
    elif Y_l[i] == 3:
        Rating.append(X_testing.iloc[i])
    elif Y_l[i] == 4:
        UserExperience.append(X_testing.iloc[i])


bug_keyword = {}
feature_keyword = {}
rating_keyword = {}
userexperience_keyword = {}

stops = set(stopwords.words('english'))
stops.add('I')
stops.add('')
stops.add('The')
stops.add('This')

for i in Bug:
    words = i.split(' ')
    for j in words:
        if j not in stops:
            if (j not in bug_keyword):
                bug_keyword[j] = 1
            else:
                bug_keyword[j]+=1



for i in Feature:
    words = i.split(' ')
    for j in words:
        if j not in stops:
            if (j not in feature_keyword):
                feature_keyword[j] = 1
            else:
                feature_keyword[j]+=1

for i in Rating:
    words = i.split(' ')
    for j in words:
        if j not in stops:
            if (j not in rating_keyword):
                rating_keyword[j] = 1
            else:
                rating_keyword[j]+=1

for i in UserExperience:
    words = i.split(' ')
    for j in words:
        if j not in stops:
            if (j not in userexperience_keyword):
                userexperience_keyword[j] = 1
            else:
                userexperience_keyword[j]+=1


Bug = sorted(bug_keyword.items(), key=lambda t: t[1])
Feature = sorted(feature_keyword.items(), key=lambda t: t[1])
Rating = sorted(rating_keyword.items(), key=lambda t: t[1])
UserExperience = sorted(userexperience_keyword.items(), key=lambda t: t[1])

j = 20
print("Top " + str(j)+ " keywords for Bugs Reviews : ")
for i in range(1,j+1):
    print(Bug[len(Bug)-i][0])

print()
print("Top " + str(j)+ " keywords for Feature Reviews : ")
for i in range(1,j+1):
    print(Feature[len(Feature)-i][0])
print()
print("Top " + str(j)+ " keywords for Rating Reviews : ")
for i in range(1,j+1):
    print(Rating[len(Rating)-i][0])
print()
print("Top " + str(j)+ " keywords for UserExperience Reviews : ")
for i in range(1,j+1):
    print(UserExperience[len(UserExperience)-i][0])





