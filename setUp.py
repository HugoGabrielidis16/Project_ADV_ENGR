
import pandas as pd
import numpy as np
import regex as re
import nltk
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.metrics.pairwise import cosine_similarity

dataframe_reviews = pd.read_csv("Database/reviews.csv",sep =";")
dataframe_overrall = pd.read_csv("Database/overall.csv",sep =";")
dataframe_sentences = pd.read_csv("Database/sentences.csv",sep =";")
dataframe_golden_set_HLT=pd.read_csv("Database/golden_set_HLT.csv",sep =";")



bug_keywords = ["bug", "fix", "problem", "issue", "defect", "crash","error","issue"]
featurerequests_keywods = ["add", "please", "could", "would", "hope", "improve", "miss", "need","prefer",
"request", "should", "suggest", "want", "wish"]




def cosine_similarity_dataframe_keywords(data_frame,keywords):
    tfidf = TfidfVectorizer(decode_error='replace', encoding='utf-8')

    for i in range(len(data_frame['review'].values)):
            data_frame['review'].values[i] = data_frame['review'].values[i].lower()
    
    dt = tfidf.fit_transform(data_frame["review"].values.astype('U'))
    made_up = tfidf.transform(keywords)

    sim = cosine_similarity(made_up, dt)
    return sim


def affichage_top_R(dataframe,kewyords,R):
    sim = cosine_similarity_dataframe_keywords(dataframe,kewyords)

    ordered_list = dataframe.iloc[np.argsort(sim[0])[::-1][0:R]][["review"]]

    for i in range(R):
        print(str(i+1) + " : " + str(ordered_list["review"].values[i]))