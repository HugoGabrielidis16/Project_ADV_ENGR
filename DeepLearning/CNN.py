
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



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

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
        df["label"][i] = 0
    elif df["label"][i] == "Feature":
        df["label"][i] = 1
    elif df["label"][i] == "Rating":
        df["label"][i] = 2
    elif df["label"][i] == "UserExperience":
        df["label"][i] = 3

sentences = df['review'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, df["label"], test_size=0.25, random_state=1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

maxlen = 100

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

y_train = np.asarray(y_train).astype('int')
y_test = np.asarray(y_test).astype('int')  

Y_train = []
for i in range(len(y_train)):
    if y_train[i] == 1:
        Y_train.append([1,0,0,0])
    elif y_train[i] == 2:
        Y_train.append([0,1,0,0])
    elif y_train[i] == 3:
        Y_train.append([0,0,1,0])
    else :
         Y_train.append([0,0,0,1])

Y_test = []
for i in range(len(y_test)):
    if y_test[i] == 1:
        Y_test.append([1,0,0,0])
    elif y_test[i] == 2:
        Y_test.append([0,1,0,0])
    elif y_test[i] == 3:
        Y_test.append([0,0,1,0])
    else :
         Y_test.append([0,0,0,1])

Y_train = np.asarray(Y_train).astype('int')
Y_test = np.asarray(Y_test).astype('int') 
print(Y_train)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index





embedding_dim = 100
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(4, activation='sigmoid'))
model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['acc'])
model.summary() 

history = model.fit(X_train, Y_train,
                    epochs=20,
                    verbose=True,
                    validation_data=(X_test, Y_test),
                    batch_size=10)


print(model.predict(X_test))

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
plt.show()