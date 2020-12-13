import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler


# clean up; remove obviously irrelevant features
with open("data/train_shuffled_all.csv", "r") as f:
    reader = csv.reader(f)
    columns = next(reader)

exclude = ["analysis_url", "id", "track_href", "type", "uri", "duration_ms", "tempo", "time_signature", "mode", "liveness", "key"]
#exclude = ["analysis_url", "id", "track_href", "type", "uri"]
columns = [e for e in columns if e not in exclude]
include = ["acousticness", "energy", "instrumentalness", "loudness", "valence", "danceability", "genre"]

data = pd.read_csv("data/train_shuffled_all.csv", usecols=include)
print(data["genre"].value_counts())
# transform categorical target variable to int
le = LabelEncoder()
le.fit(data.genre)
data["genre"] = le.transform(data.genre)

X = data.iloc[:,data.columns != "genre"]  # independent feature columns
y = data.iloc[:,-1]    # target column i.e genre

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X, columns=X.columns)
scaled_df = scaled_df.assign(genre=y)
scaled_df.to_csv("data/preprocessed.csv", index=False)