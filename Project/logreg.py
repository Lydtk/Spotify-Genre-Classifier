import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.dummy import DummyClassifier
import csv
from sklearn.preprocessing import PolynomialFeatures, label_binarize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score, classification_report
from sklearn.metrics import confusion_matrix

def cross_val_C():
    C_range = [0.01, 0.1, 0.5, 1, 5, 10, 50]
    scores = []
    kf = model_selection.KFold(n_splits=10)
    for C in C_range:
        tmp = []
        for train_idx, test_idx in kf.split(X):
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_test = y[test_idx]
            y_train = y[train_idx]
            lm = linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs', C=C, penalty="l2")
            lm.fit(x_train, y_train)
            y_pred = lm.predict(x_test)
            tmp.append(f1_score(y_test, y_pred, average="micro"))
    
        scores.append(np.mean(tmp))
    plt.plot(C_range, scores)
    plt.title("F1 Scores vs. C")
    plt.xlabel("C_i"); plt.ylabel("F1 score")
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

def cross_val_q():
    kf = model_selection.KFold(n_splits=5)
    q_range = [1,2,3,4,5,6,7,8,9,10]
    scores = []
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        model = linear_model.LogisticRegression(multi_class='ovr', dual=False, solver='lbfgs', C=0.4, penalty="l2", max_iter=10000)
        temp = []
        for train, test in kf.split(Xpoly):
            y_test = y[test]
            y_train = y[train]
            model.fit(Xpoly[train], y_train)
            y_pred = model.predict(Xpoly[test])
            temp.append(f1_score(y_test, y_pred, average="micro"))

        scores.append(np.mean(temp))
    plt.plot(q_range, scores)
    plt.title("F1 Scores vs. q")
    plt.xlabel("q_i"); plt.ylabel("F1 score")
    plt.show()

def cf_matrix():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2)
    lm = linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs', penalty="l2", C=10)
    lm.fit(X_train, y_train)
    y_pred = lm.decision_function(X_test)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Confusion Matrix')
    metrics.plot_confusion_matrix(lm, X_test, y_test, ax = ax)
    target_names = ["Classical", "Hip-Hop", "Rock"]
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n BASELINE")
    dummy = DummyClassifier(strategy='uniform').fit(X_train, y_train)  #train dummy
    ydummy = dummy.predict(X_test) #predict baseline
    print(classification_report(y_test, ydummy, target_names=target_names))
    plt.show()

def plot_multiclass_roc(n_classes, figsize=(17, 6)):
    clf = linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs', penalty="l2", C=0.4)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2)
    clf.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    
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
#print(data.genre)
# ["hiphop", "rock", "classical"] = [1, 2, 0]

X = data.iloc[:,data.columns != "genre"]  # independent feature columns
y = data.iloc[:,-1]    # target column i.e genre

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X, columns=X.columns)
X = scaled_df.iloc[:,data.columns != "genre"]  # independent feature columns

# plot_multiclass_roc(n_classes=["classical", "hiphop", "rock"], figsize=(16, 10))
#cf_matrix()
cross_val_C()
#cross_val_q()



