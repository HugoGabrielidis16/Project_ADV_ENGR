import pandas as pd
import numpy as np



# Retrieve database

dataframe_reviews = pd.read_csv("Database/reviews.csv",sep =";")
dataframe_overrall = pd.read_csv("Database/overall.csv",sep =";")
dataframe_sentences = pd.read_csv("Database/sentences.csv",sep =";")
dataframe_golden_set_HLT=pd.read_csv("Database/golden_set_HLT.csv",sep =";")




