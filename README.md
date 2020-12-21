# Spotify-Genre-Classifier

+ [knn.py](Project/knn.py) - Houses code for all things related to KNN Classifier(Cross-val, training, plots etc).
  
+ [logreg.py](Project/logreg.py) - Houses code for all things related to Logistic Regression Classifier(Cross-val, training, plots etc).

+ [get_data.py](Project/get_data.py) - Used to fetch track info using the Spotify API and place inside a pandas dataframe, which is finally converted to a CSV file.
  
+ [produce_trainready.py](Project/produce_trainready.py) - Preprocessing/Normalization of the CSV obtained by `get_data.py`.

+ [feature_select.py](Project/feature_select.py) - Used to analyse importance of features in dataset using ANOVA F-scores.