import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import itertools


def readData(textfile):
    data = []
    with open(textfile, 'r', encoding='utf8') as fin:
        for line in fin:
            split = line.split("\t")
            split[0] = re.sub(r'\d+', '', split[0]).strip()
            split[0] = re.sub(r'[^\w\s]', '', split[0])
            split[1] = split[1].strip()
            data.append(split)
    return data


def pipeLine(feat_comb, classifier="RF"):
    if classifier == "RF":
        clf = RandomForestClassifier()
        pipeline = Pipeline([("features", feat_comb), ("rf", clf)])

    elif classifier == "DT":
        clf = DecisionTreeClassifier()
        pipeline = Pipeline([("features", feat_comb), ("dt", clf)])

    elif classifier == "NB":
        clf = BernoulliNB()
        pipeline = Pipeline([("features", feat_comb), ("nb", clf)])

    elif classifier == "SVM":
        clf = SVC(kernel="linear")
        pipeline = Pipeline([("features", feat_comb), ("svm", clf)])

    return pipeline

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def trainAndEvaluate(pipeline, clf_name, sparse_data, labels, split=0.33):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(sparse_data, labels,
                                                        test_size=split,
                                                        random_state=42)
    print("\ndone splitting data ... ")
    print("\ncombing features and training ... ")
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    print("\nAccuracy score for %s classifier : \n" % clf_name, accuracy_score(y_test, predicted))
    cm = confusion_matrix(y_test, predicted)
    classes = y_train.unique()
    plot_confusion_matrix(cm=cm, classes=classes)

def gridSearch(pipeline, param_grid, sparse_data, labels, clf_name):
    X_train, X_test, y_train, y_test = train_test_split(sparse_data, labels,
                                                        test_size=0.33,
                                                        random_state=42)
    # grid search
    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               verbose=10)
    grid_search.fit(X_train, y_train)
    print("\n Best configuration for %s after grid search have the following parameter: \n" % clf_name,
          grid_search.best_estimator_)

# import the data
data = readData("Assignment 6/train.txt")
data = np.array(data).astype("str")
data = pd.DataFrame(data)
print("\nSize of data: ", data.shape)

# feature extraction step
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(data[0])
print("\nSize of vectorized data (CountVectorizer): ", x_train.shape)

# PCA (svd) on the data
svd = TruncatedSVD(n_components=20)
# chisquare
selection = SelectKBest(chi2, k=10)
# combine the features
combined_features = FeatureUnion([("svd", svd), ("chi_sq", selection)])

# Decision tree
pipeline = pipeLine(feat_comb=combined_features, classifier="DT")
trainAndEvaluate(pipeline=pipeline, clf_name="Decision Tree",
                 sparse_data=x_train, labels=data[1])

# random forest
pipeline = pipeLine(feat_comb=combined_features, classifier="RF")
trainAndEvaluate(pipeline=pipeline, clf_name="Random Forest",
                 sparse_data=x_train, labels=data[1])

# Naive Bayes
pipeline = pipeLine(feat_comb=combined_features, classifier="NB")
trainAndEvaluate(pipeline=pipeline, clf_name="Naive Bayes",
                 sparse_data=x_train, labels=data[1])

# for SVM we will train and test on a fraction of data(~10,000 rows) because of classifier's limit
sampled = data.sample(n=10000)
sampled_sparse = vectorizer.fit_transform(sampled[0])
pipeline = pipeLine(feat_comb=combined_features, classifier="SVM")
trainAndEvaluate(pipeline=pipeline, clf_name="SVM",
                 sparse_data=sampled_sparse, labels=sampled[1])

# grid search parameters
param_grid = dict(features__svd__n_components=[10, 20, 30],
                  features__chi_sq__k=[10, 20, 30])

# grid search with Decision tree
pipeline = pipeLine(feat_comb=combined_features, classifier="DT")
gridSearch(pipeline=pipeline, param_grid=param_grid, sparse_data=x_train,
           labels=data[1], clf_name= "Decision Tree")