
from sklearn.datasets import make_multilabel_classification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import json
from prepare_json import DataFrame_json_label_Matrix_test, DataFrame_json_review_Matrix_test

import tensorflow as tf
import prepare_json 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
with open("json/all.json", "r") as read_file:
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



countv = CountVectorizer(min_df = 5, ngram_range=(1,1), stop_words="english")
X_train_tf = countv.fit_transform(X_train)

# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold

import tensorflow as tf


from sklearn.metrics import accuracy_score


# get the model
def get_model(n_inputs, n_outputs):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results

# load dataset
# evaluate model

#print(DataFrame_json_label_Matrix_test.shape[1])
""" results = evaluate_model(DataFrame_json_review_Matrix_test, DataFrame_json_label_Matrix_test)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
 """

l = []

for i in range(len(DataFrame_json_label_Matrix_test)):
    h = [0,0,0,0]
    h[int(DataFrame_json_label_Matrix_test[i])-1] = 1
    l.append(h)
l = np.array(l)

results = evaluate_model(DataFrame_json_review_Matrix_test, l)
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))