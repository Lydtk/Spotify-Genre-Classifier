import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv

from sklearn.utils.validation import check_random_state

def find_k(X,y):
    kf = KFold(n_splits=5)
    f1Scores = []
    k_range = list(range(1, 31))     #for cross-val
    # k_range = [9]                        #after choosing best k

    # for k in k_range:
    #     model =  KNeighborsClassifier(n_neighbors = k) #passing in different number of neighbours
    #     temp = []
    #     for train, test in kf.split(X):       #different combinations of splitting test/train according to k folds
    #         model.fit(X[train], y[train])
    #         ypred = model.predict(X[test])     #predicted y according to model
    #         temp.append(f1_score(y[test], ypred, average='micro'))

    #     f1Scores.append(np.array(temp).mean())  

    knn = KNeighborsClassifier()
    param_grid = dict(n_neighbors=k_range)
    param_grid["weights"] = ["uniform", "distance"]
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)
    # Dictionary containing the parameters (k) used to generate that score
    print(grid.best_params_)
    print(grid.best_estimator_)
    # print("Max Score",max(f1Scores))       # finding highest f1-best precision
    # plt.plot(k_range, f1Scores, linewidth=2)   #score bar
    # plt.xlabel('Number of Neighbours')
    # plt.ylabel('F1 score')
    # plt.title("f1 score at different K values ")
    # plt.show()

def baseline(X,y):
    dummy = DummyClassifier(strategy='uniform').fit(X, y)  #train dummy
    ydummy = dummy.predict(X) #predict baseline
    return ydummy

def cf_matrix():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Confusion Matrix')
    plot_confusion_matrix(knn, X_test, y_test, ax = ax)
    target_names = ["Classical", "Hip-Hop", "Rock"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    plt.show()

def compare_train_test():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=3)
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    print("TRAIN\n")
    print(classification_report(y_train, preds))
    print(confusion_matrix(y_train,preds))
    
    preds = model.predict(X_test)
    print("TEST")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test,preds))


def plot_multiclass_roc(n_classes, figsize=(17, 6)):
    clf = KNeighborsClassifier(n_neighbors=9)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=3)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(len(n_classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    for i in range(len(n_classes)):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], n_classes[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

with open("data/train_shuffled_all.csv", "r") as f:
    reader = csv.reader(f)
    columns = next(reader)

exclude = ["analysis_url", "id", "track_href", "type", "uri", "duration_ms", "tempo", "time_signature", "mode", "liveness", "key"]
#exclude = ["analysis_url", "id", "track_href", "type", "uri"]
columns_overfit = [e for e in columns if e not in exclude]
include = ["acousticness", "energy", "instrumentalness", "loudness", "valence", "danceability", "genre"]

data = pd.read_csv("data/train_shuffled_all.csv", usecols=include)
print(data["genre"].value_counts())
# transform categorical target variable to int
le = LabelEncoder()
le.fit(data.genre)
data["genre"] = le.transform(data.genre)
# ["hiphop", "rock", "classical"] = [1, 2, 0]
X = data.iloc[:,data.columns != "genre"]  # independent feature columns
y = data.iloc[:,-1]    # target column i.e genre

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X, columns=X.columns)
X = scaled_df.iloc[:,data.columns != "genre"]  # independent feature columns

# cross val to find k -number of neighbours
find_k(X, y)

plot_multiclass_roc(n_classes=["classical", "hiphop", "rock"], figsize=(16, 10))
cf_matrix()
compare_train_test()


