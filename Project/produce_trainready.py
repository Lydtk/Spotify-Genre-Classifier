import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# clean up; remove obviously irrelevant features
with open("data/train_shuffled_all.csv", "r") as f:
    reader = csv.reader(f)
    columns = next(reader)

exclude = ["analysis_url", "id", "track_href", "type", "uri", "duration_ms", "tempo", "time_signature", "mode", "liveness", "key"]
#exclude = ["analysis_url", "id", "track_href", "type", "uri"]
columns = [e for e in columns if e not in exclude]
apply_transform = ["acousticness", "energy", "instrumentalness", "loudness", "valence", "danceability"]
include = ["acousticness", "energy", "instrumentalness", "loudness", "valence", "danceability", "genre"]

data = pd.read_csv("data/train_shuffled_all.csv", usecols=include)
print(data["genre"].value_counts())

# transform categorical target variable to int
le = LabelEncoder()
le.fit(data.genre)
data["genre"] = le.transform(data.genre)

scaler = MinMaxScaler()

data[apply_transform] = scaler.fit_transform(data[apply_transform])
data.to_csv("data/preprocessed.csv", index=False)