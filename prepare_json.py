import json
import pandas as pd
from setUp import *
import nltk
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open("json/all.json", "r") as read_file:
    data = json.load(read_file)



Review_label = []




for i in range(len(data)):
    a = data[i]["comment"]
    b = data[i]["label"]
    Review_label.append([a,b])



DataFrame_json = pd.DataFrame(Review_label)
DataFrame_json.columns = ["review","label"]



DataFrame_json_review_Matrix = np.transpose(cosine_similarity_dataframe_keywords(DataFrame_json,bug_keywords,"review"))

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
    
