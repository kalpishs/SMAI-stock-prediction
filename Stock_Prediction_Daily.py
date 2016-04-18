
# coding: utf-8

# In[1]:

import numpy
import csv
import urllib
from sklearn import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

data = list()
with open('E:\Lecs\IIIT\SMAI\Project\Data\\table.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
data = numpy.array(data)
data = data[1:, 1:]
data = data.astype(float)
labels = ((data[:, 3] - data[:, 0]) > 0).astype(int)
print labels


# In[3]:

def extract_features(data, indices):
    data2 = data[1:, :]
    features = data[:-1] - data2
    return features[:, indices]


# In[4]:

features = extract_features(data, [1, 2])
train_features = features[:1000]
test_features = features[1000:]
train_labels = labels[:1000]
test_labels = labels[1000:-1]


# In[43]:

clf = svm.SVC(kernel = 'rbf', C = 1.2, gamma = 0.001)
clf.fit(train_features, train_labels)


# In[44]:

predicted = clf.predict(test_features)
print accuracy_score(test_labels, predicted)
print recall_score(test_labels, predicted)
print precision_score(test_labels, predicted)


# In[61]:

step = numpy.arange(0, len(test_labels), 0.2)
plt.step(step, numpy.repeat(test_labels, 5))
#plt.plot(plot_predicted)


# In[ ]:



