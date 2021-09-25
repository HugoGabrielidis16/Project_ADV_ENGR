import pandas as pd
import numpy as np
import regex as re
import nltk
import re
import html

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.metrics.pairwise import cosine_similarity

#nltk.download('wordnet')

# Retrieve database

dataframe_reviews = pd.read_csv("Database/reviews.csv",sep =";")
dataframe_overrall = pd.read_csv("Database/overall.csv",sep =";")
dataframe_sentences = pd.read_csv("Database/sentences.csv",sep =";")
dataframe_golden_set_HLT=pd.read_csv("Database/golden_set_HLT.csv",sep =";")


def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


#dataframe_reviews['tokens'] = dataframe_reviews['review'].map(lambda x : nltk.tokenize.word_tokenize(x))
#dataframe_overrall['tokens'] = dataframe_overrall['review'].map(lambda  x: nltk.tokenize.word_tokenize(x))
#dataframe_sentences['tokens'] = dataframe_sentences['review'].map(lambda x : nltk.tokenize.word_tokenize(x))
#dataframe_golden_set_HLT['tokens'] = dataframe_golden_set_HLT['review'].map(lambda x: nltk.tokenize.word_tokenize(x))


#print(dataframe_overrall['tokens'])

#lemmatizer = nltk.stem.WordNetLemmatizer()

#print(lemmatizer.lemmatize("Better".lower()))


tfidf = TfidfVectorizer(decode_error='replace', encoding='utf-8')

for i in range(len(dataframe_overrall['review'].values)):
        dataframe_overrall['review'].values[i] = dataframe_overrall['review'].values[i].lower()



dt = tfidf.fit_transform(dataframe_overrall["review"].values.astype('U'))
text = "crash"
made_up = tfidf.transform([text])

sim = cosine_similarity(made_up, dt)
print()
print('The review that have the better cosine similarity with the words "'+text+ '" are : ')

ordered_list = dataframe_overrall.iloc[np.argsort(sim[0])[::-1][1:6]][["review"]]

print(ordered_list["review"].values)
 