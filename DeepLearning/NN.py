
from sklearn.datasets import make_multilabel_classification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json

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
        df["label"][i] = 1
    elif df["label"][i] == "Feature":
        df["label"][i] = 2
    elif df["label"][i] == "Rating":
        df["label"][i] = 3
    elif df["label"][i] == "UserExperience":
        df["label"][i] = 4

X_train, X_test, Y_train, Y_test = train_test_split(df['review'],
                                                    df['label'],
                                                    test_size=0.4,
                                                    stratify=df['label'])

countv = TfidfVectorizer(min_df = 2, ngram_range=(1,8), stop_words="english")
X_train_tf = countv.fit_transform(X_train)
X_train_tf = X_train_tf.toarray() 


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    #tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    #tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(32, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                    ])


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_tf, Y_train, epochs=10)

X_test_tf = countv.transform(X_test)
X_test_tf = X_test_tf.toarray()

model.evaluate(X_test_tf, Y_test)