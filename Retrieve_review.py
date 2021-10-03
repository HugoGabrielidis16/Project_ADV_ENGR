import pandas as pd
import numpy as np
import regex as re
import nltk
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.metrics.pairwise import cosine_similarity


from function import *
from setUp import *

#nltk.download('wordnet')

# Retrieve database






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





affichage_top_R(dataframe_overrall,bug_keywords,20,"review")
 