import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

def main():
    #Reading in Dataset 
    dataframe = pd.read_csv('data/train_shuffled_all.csv' , usecols = ['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','genre'])
    
    #To plot data for 2 features
    # data_pl = {'Dancebility': X1, 'energy': X2, 'Genre': y}
    # sns.scatterplot(x='Dancebility', y='energy', hue='Genre',data=pd.DataFrame(data_pl), palette='dark')
    # plt.title("Plot of the Data")
    # plt.show()

     #Replacing text outputs with numbers
    dataframe.replace(to_replace ="Rock",  
                 value = -1,  
                  inplace = True) 
    dataframe.replace(to_replace ="Hip-Hop",  
                    value = 0,  
                    inplace = True) 
    dataframe.replace(to_replace ="Classical",  
                    value = 1,  
                    inplace = True) 
    dataframe.to_csv('output.csv',  
                 index = False)
    #new csv
    data= pd.read_csv('output.csv' , usecols = ['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','genre'])
    print(data.head())

    #features
    XAcst= data.iloc[:,0]
    XDnce = data.iloc[:,1]
    XDur = data.iloc[:,2]
    XEnrg = data.iloc[:,3]
    XInst = data.iloc[:,4]
    XKey = data.iloc[:,5]
    XLive = data.iloc[:,6]
    XLoud= data.iloc[:,7]
    XMode = data.iloc[:,8]
    XSpeech = data.iloc[:,9]
    XTmpo = data.iloc[:,10]
    XTimSig = data.iloc[:,11]
    XVal = data.iloc[:,12]

    #output (-1,0,1)
    y = data.iloc[:,13]
    #features
    X = np.column_stack((XAcst,XDnce,XDur,XEnrg,XInst,XKey, XLive ,XLoud, XMode, XSpeech, XTmpo, XTimSig, XVal))
    

    #cross val to find k -number of neighbours
    find_k(X,y)

    roc_curves(X,y,32)
   

#t
def find_k(X,y):

    kf = KFold(n_splits=5)
    f1Scores = []

    ydummy = baseline(X,y)               #predict baseline
    k_range = [1,10,25,30,32,35,50]     #for cross-val
    # k_range = [32]                        #after choosing best k

    for k in k_range:

        model =  KNeighborsClassifier(n_neighbors = k, weights='uniform') #passing in different number of neighbours
        temp = []
        for train, test in kf.split(X):       #different combinations of splitting test/train according to k folds
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])     #predicted y according to model
            temp.append(f1_score(y[test], ypred, average='macro'))

        #arrays for plotting
        f1Scores.append(np.array(temp).mean())     #add to array of f1 scores
        print("k:" ,k, "score: ",f1Scores)
        ypred = model.predict(X)

    #Plotting confusion matric + classif. report *** separate into func later.
    print("KNN:\n",confusion_matrix(y, ypred))
    print(classification_report(y, ypred,target_names=['Hip-Hop', 'Classical', 'Rock']))
    print("Dummy Classifier:\n",confusion_matrix(y, ydummy))
    print(classification_report(y, ydummy,target_names=['Hip-Hop', 'Classical', 'Rock']))

    #Plotting predicted results vs real results
    print("Max Score",max(f1Scores))       # finding highest f1-best precision
    plt.plot(k_range, f1Scores, linewidth=2)   #score bar
    plt.xlabel('Number of Neighbours')
    plt.ylabel('F1 score')
    plt.title("f1 score at different K values ")
    plt.show()

def baseline(X,y):
    dummy = DummyClassifier(strategy='most_frequent').fit(X, y)  #train dummy
    ydummy = dummy.predict(X) #predict baseline


    return ydummy




def roc_curves(X,y,k):     # optimal parameters given--------->>>> Needs to be checked and rewritten
    # y = label_binarize(y, classes=[-1,0,1])

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit model
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=k, weights='uniform'))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}

    n_class = 3

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        
    # plotting    
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    

    # #make models
    # model_knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    # model_dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    # model_random = DummyClassifier(strategy="uniform").fit(X, y)

    # #train models
    # model_knn.fit(X, y)    #doesn't need Xpoly 

    # #Split data for training and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # #plot kNN classifier
    # model_knn.fit(X_train, y_train)
    # y_pred = model_knn.predict_proba(X_test)[::, 1]  #predicted probability of 1 and -1
    # fpr, tpr, _ = roc_curve(y_test, y_pred)         #for plotting
    # plt.plot(fpr, tpr, label="kNN Classifier")

    # #plot dummy baseline
    # model_dummy.fit(X_train, y_train)
    # y_pred = model_dummy.predict_proba(X_test)[::, 1]
    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # plt.plot(fpr, tpr, label="Baseline: Most Frequent")

    # #plot dummy baseline -- random
    # model_random.fit(X_train, y_train)
    # y_pred = model_random.predict_proba(X_test)[::,1]
    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # plt.plot(fpr, tpr, label="Baseline: Random", ls="--")

    # #plot labels
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive rate")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
