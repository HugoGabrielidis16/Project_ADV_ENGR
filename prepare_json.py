import json
import pandas as pd
from setUp import *
import nltk
import warnings
from function import *
warnings.filterwarnings("ignore", category=UserWarning)

with open("json/all.json", "r") as read_file:
    data = json.load(read_file)



Review_label = []




for i in range(len(data)):
    a = data[i]["comment"]
    b = data[i]["label"]
    Review_label.append([a,b])


print(len(data))
DataFrame_json = pd.DataFrame(Review_label)
DataFrame_json.columns = ["review","label"]

tfidf_word = TfidfVectorizer(stop_words=stopwords, min_df=1000)
#print(DataFrame_json["review"])
dt_word = tfidf_word.fit_transform(DataFrame_json["review"])
#print(dt_word)
#DataFrame_json_review_Matrix = np.transpose(cosine_similarity_dataframe_keywords(DataFrame_json,bug_keywords+featurerequests_keywods,"review"))

#print(dt_word.T)
DataFrame_json_review_Matrix = cosine_similarity(dt_word, dt_word)

#affichage(DataFrame_json_review_Matrix.shape)


DataFrame_json_label_Matrix = np.transpose(np.zeros(len(DataFrame_json["label"])))


for i in range(len(DataFrame_json_label_Matrix)):
    if DataFrame_json["label"][i] == "Bug":
        DataFrame_json_label_Matrix[i] = 1
    elif DataFrame_json["label"][i] == "Feature":
        DataFrame_json_label_Matrix[i] = 2
    elif DataFrame_json["label"][i] == "Rating":
        DataFrame_json_label_Matrix[i] = 3
    elif DataFrame_json["label"][i] == "UserExperience":
        DataFrame_json_label_Matrix[i] = 4
    """ else:
        DataFrame_json_label_Matrix[i] = 0 """
    
print(len(DataFrame_json))
DataFrame_json_label_Matrix_test = DataFrame_json_label_Matrix[0:3670]
DataFrame_json_review_Matrix_test = DataFrame_json_review_Matrix[0:3670]



