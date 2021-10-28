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

countv = CountVectorizer(min_df = 5, ngram_range=(1,1), stop_words="english")
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


for i in range(len(X_testing)):
    if Y_l[i] =="Bug":
        print(X_testing.iloc[i])
        print(Y_l[i])


