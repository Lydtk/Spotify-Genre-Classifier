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
import csv
from sklearn.utils.validation import check_random_state

#To find k: the optimal number of neighbours to check
def find_k(X,y):
    kf = KFold(n_splits=5)
    f1Scores = []
    k_range = list(range(1, 50))     #for cross-val

    for k in k_range:
        model =  KNeighborsClassifier(n_neighbors = k) #passing in different number of neighbours
        temp = []
        for train_idx, test_idx in kf.split(X):       #different combinations of splitting test/train according to k folds
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_test = y[test_idx]
            y_train = y[train_idx]
            model.fit(x_train, y_train)
            ypred = model.predict(x_test)     #predicted y according to model
            temp.append(f1_score(y_test, ypred, average='micro'))

        f1Scores.append(np.array(temp).mean())#getting average f1 score of all the train test combos

    #plotting f1 chart
    plt.figure(figsize=(10,6))
    plt.plot(k_range,f1Scores,color = 'blue',linestyle='dashed',
            marker='o',markerfacecolor='red', markersize=10)
    plt.title('F1 Score vs. K Value')
    plt.xlabel('K')
    plt.ylabel('F1 Score')
    plt.show()


def cf_matrix():
    #split into train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

    #Predict with knn model with optimal k
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    #Predict with dummy model
    dummy = DummyClassifier(strategy='uniform').fit(X_train, y_train)  #train dummy
    ydummy = dummy.predict(X_test) #predict baseline


    #Plot Confusion Matrix -knn
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Confusion Matrix - KNN')
    plot_confusion_matrix(knn, X_test, y_test, ax = ax)
    target_names = ["Classical", "Hip-Hop", "Rock"]
    print("Knn Classification Report")
    print(classification_report(y_test, y_pred, target_names=target_names))
    plt.show()


    #Plot Confusion Matrix -Dummy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Confusion Matrix - Dummy Classifier')
    plot_confusion_matrix(dummy, X_test, y_test, ax = ax)
    target_names = ["Classical", "Hip-Hop", "Rock"]
    print("Dummy Classification Report")
    print(classification_report(y_test, ydummy, target_names=target_names))
    plt.show()


def plot_multiclass_roc( n_classes):
    #make model
    clf = KNeighborsClassifier(n_neighbors=12)
    #split data for training test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=3)
    #"train" model
    clf.fit(X_train, y_train)
    #predicted probability of 1 and -1
    y_score = clf.predict_proba(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # baseline graph
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(len(n_classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # roc for each class
    fig, ax = plt.subplots(figsize=(10,6))
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

data = pd.read_csv("data/preprocessed.csv")
print(data.head())
# ["classical","hiphop", "rock"] = [0,1, 2]
X = data.iloc[:,data.columns != "genre"]  # independent feature columns
y = data.iloc[:,-1]    #target column i.e. genre

#To plot data for 2 features as an example
# X1 = data.iloc[:,0]
# X2 = data.iloc[:,1]
# data_pl = {'Acousticness': X1, 'Danceability': X2, 'Genre': y}
# sns.scatterplot(x='Acousticness', y='Danceability', hue='Genre',data=pd.DataFrame(data_pl), palette='dark')
# plt.title("Plot of the Data")
# plt.show()

# cross val to find k -number of neighbours
find_k(X, y)
# plot roc curve
plot_multiclass_roc( n_classes=["classical", "hiphop", "rock"])
# confusion matrices + classification reports
cf_matrix()


# not used
# def baseline(X,y):
#     dummy = DummyClassifier(strategy='uniform').fit(X, y)  #train dummy
#     ydummy = dummy.predict(X) #predict baseline
#     return ydummy
