



from scipy.sparse import data

from prepare_json import *
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

bug_keywords = ["bug", "fix", "problem", "issue", "defect", "crash","error","issue","problem","uninstalling"]



featurerequests_keywods = ["add", "please", "could", "would", "hope", "improve", "miss", "need","prefer",
"request", "should", "suggest", "want", "wish"]

user_experience_keywords = ["help","support","assist","when","situation"]

rating_keywords = ["great","good","very","cool","love","hate","bad","worst"]

y = []
for i in DataFrame_json["review"]:
    i = i.split(" ")
    Bug = 0
    Feature = 0
    Rating = 0
    UserExperience = 0
    for j in i:
        j = j.lower()
        j = lemmatizer.lemmatize(j)
        if j in bug_keywords:
            Bug+=1
        else:
            if j in featurerequests_keywods:
               Feature+=1
            else:
                if j in user_experience_keywords:
                    Rating +=1
                else:
                    if j in rating_keywords:
                        UserExperience +=1
    if Bug > 0 or Feature > 0 or Rating > 0 or UserExperience > 0:
        if max(Bug,Feature,Rating,UserExperience) == Bug:
            y.append("Bug")
        elif max(Bug,Feature,Rating,UserExperience) == Feature:
            y.append("Feature")
        elif max(Bug,Feature,Rating,UserExperience) == Rating:
            y.append("Rating")
        elif max(Bug,Feature,Rating,UserExperience) == UserExperience:
            y.append("UserExperience")
    else:
        y.append("Unclassified")



accuracy = 0


for i in range(len(y)):

    if y[i] == DataFrame_json["label"][i]:
        accuracy+=1

accuracy = accuracy/len(y)

print("Accuracy = " + str(accuracy))

