import pandas as pd
import numpy as np

df = pd.read_csv('pcgarage.csv', delimiter = '\t', encoding = 'utf-16')

#print(df)

df_corpus = df.iloc[:, -3:]

#print(df_corpus)

df_corpus["corpus"] = df_corpus["pro"].astype(str) + " " + \
                        df_corpus["contra"].astype(str) + " " + \
                        df_corpus["altele"].astype(str)


data = pd.DataFrame (tokens)
i = 1
for rows in np.array_split(data, 35):
    rows.to_csv("inputuri/pc" + str(i) + ".csv", encoding = "UTF-8", index = False)
    i = i + 1
