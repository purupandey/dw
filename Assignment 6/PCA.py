import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

data = readData("Assignment 6/test-gold.txt")
data = np.array(data).astype("str")
len(data)

# feature extraction step
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(data[:,0])
print(x_train)

# PCA on the data
svd = TruncatedSVD(n_components=25, random_state=42)
svd_op = svd.fit_transform(x_train)

# plot data


# train test split
X_train, X_test, y_train, y_test = train_test_split(svd_op, data[:, 1], test_size=0.33, random_state=42)

# RF
rf_clf = RandomForestClassifier()
rf_clf.fit(X=X_train, y=y_train)

predictions = rf_clf.predict(X= X_test)
accuracy_score(y_test, predictions)

# LR
lr_clf = LogisticRegression()
lr_clf.fit(X=X_train, y=y_train)

predictions = lr_clf.predict(X= X_test)
accuracy_score(y_test, predictions)