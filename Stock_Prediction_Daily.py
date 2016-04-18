
# coding: utf-8

# In[1]:

import numpy
import csv
import urllib
from sklearn import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
from pybrain.datasets import *
from pybrain.structure.modules import *
#%matplotlib inline


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


# In[35]:

def t_high(t, X):
    return max(X[:-t])


# In[36]:

def t_low(t, X):
    return min(X[:-t])


# In[37]:

def volume_high(t, X):
    return max(X[:-t])


# In[38]:

def volume_low(t, X):
    return min(X[:-t])


# In[95]:

def extract_features(data, indices):
    data = data[:, [0, 1, 2, 3, 5]]
    data2 = data[1:, :]
    features = data[:-1] - data2
    Phigh = t_high(5, data[:, 1])
    Plow = t_low(5, data[:, 2])
    vhigh = volume_high(5, data[:, 4])
    vlow = volume_low(5, data[:, 4])
    Odiff_by_highlow = features[:, 0]/ float(Phigh - Plow)
    Cdiff_by_highlow = features[:, 1]/float(Phigh - Plow)
    mov_avg_by_data = list()
    for i in range(len(features)):
        mov_avg_by_data.append(numpy.mean(data[:i+1, :], axis = 0)/data[i, :])
    mov_avg_by_data = numpy.array(mov_avg_by_data)
    features = numpy.column_stack((features, Odiff_by_highlow, Cdiff_by_highlow, mov_avg_by_data))
    print numpy.shape(features)
    return features[:, indices]


# In[97]:

features = extract_features(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
train_features = features[:1000]
test_features = features[1000:]
train_labels = labels[:1000]
test_labels = labels[1000:-1]


# In[100]:

clf = svm.SVC(kernel = 'rbf', C = 1.2, gamma = 0.001)
clf.fit(train_features, train_labels)


# In[101]:

predicted = clf.predict(test_features)
print accuracy_score(test_labels, predicted)
print recall_score(test_labels, predicted)
print precision_score(test_labels, predicted)


# In[102]:


step = numpy.arange(0, len(test_labels))
plt.subplot(211)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.plot(step, test_labels, drawstyle = 'step')
plt.subplot(212)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.plot(step, predicted, drawstyle = 'step')
plt.show()
#plt.plot(plot_predicted)


# In[103]:

#net = RecurrentNetwork()
#net.addInputModule(LinearLayer(3, name = 'in'))
#net.addInputModule(SigmoidLayer(4, name = 'hidden'))
#net.addOutputModule(LinearLayer(1, name = 'output'))
#net.addConnection(FullConnection(net['in'], net['hidden'], name = 'c1'))
#net.addConnection(FullConnection(net['hidden'], net['output'], name = 'c2'))
#net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))
net = buildNetwork(12, 20, 1, hiddenclass = LSTMLayer, outclass = SigmoidLayer, recurrent = True)
ds = ClassificationDataSet(12, 1)
for i, j in zip(train_features, train_labels):
    ds.addSample(i, j)


# In[104]:

trainer = BackpropTrainer(net, ds)


# In[105]:

epochs = 10
for i in range(epochs):
    trainer.train()


# In[106]:

predicted = list()
for i in test_features:
    #print net.activate(i)
    predicted.append(int(net.activate(i)>0.5))
predicted = numpy.array(predicted)


# In[107]:

print accuracy_score(test_labels, predicted)
print recall_score(test_labels, predicted)
print precision_score(test_labels, predicted)


# In[108]:

step = numpy.arange(0, len(test_labels))
plt.subplot(211)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.plot(step, test_labels, drawstyle = 'step')
plt.subplot(212)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.plot(step, predicted, drawstyle = 'step')
plt.show()
#plt.plot(plot_predicted)


# In[ ]:



