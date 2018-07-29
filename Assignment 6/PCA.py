import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)

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

data = readData("Assignment 6/train.txt")
data = np.array(data).astype("str")
len(data)

# for every language select 300 documents
data = pd.DataFrame(data)
languages = data[1].unique()
sample_df = pd.DataFrame([])
for i in languages:
    sample_df = sample_df.append(data[data[1]==i].sample(300))


# feature extraction step
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(sample_df[0])
print(x_train.shape)

# PCA on the data
svd = TruncatedSVD(n_components=10, random_state=42)
svd_op = svd.fit_transform(x_train)

# plot data
sns.pairplot(pd.DataFrame(svd_op))
plt.show()
