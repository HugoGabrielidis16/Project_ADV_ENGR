
import matplotlib.pyplot as plt
import numpy as np


from setUp import *



sim = cosine_similarity_dataframe_keywords(dataframe_overrall,bug_keywords)


data = []
for i in range(len(sim)):
    data.append(sim[i])
