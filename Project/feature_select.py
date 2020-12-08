import numpy as np 
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv

# clean up; remove obviously irrelevant features
with open("data/song_data_train.csv", "r") as f:
    reader = csv.reader(f)
    columns = next(reader)

exclude = ["analysis_url", "id", "track_href", "type", "uri"]
columns = [e for e in columns if e not in exclude]

data = pd.read_csv("data/train_shuffled_all.csv", usecols=columns)
# transform categorical target variable to int
le = LabelEncoder()
le.fit(data.genre)
data["genre"] = le.transform(data.genre)

X = data.iloc[:,data.columns != "genre"]  # independent feature columns
X_normal = StandardScaler().fit_transform(X)
y = data.iloc[:,-1]    # target column i.e genre

print("Normalised: ANOVA F value score")
bestfeatures = SelectKBest(score_func=f_classif, k=13)
fit = bestfeatures.fit(X_normal, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  # naming the dataframe columns
print(featureScores.nlargest(13,'Score'))  # print 10 best features