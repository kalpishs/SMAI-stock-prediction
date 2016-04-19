
# coding: utf-8

# In[131]:

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


# In[132]:

def multiple_days_forward(data, days):
    labels = ((data[days:, 3] - data[days:, 0]) > 0).astype(int)
    data = data[:-days, :]
    return data, labels


# In[155]:

data = list()
print "Enter Company/Stock: "
print "1. Nifty"
print "2. TCS"
print "3. HCL"
print "4. Infy"
print "5. ONGC"
print "6. Reliance"
case = int(input())


# In[156]:

if case == 1:
    url = 'E:\Lecs\IIIT\SMAI\Project\Data\\Nifty.csv'
elif case == 2:
    url = 'E:\Lecs\IIIT\SMAI\Project\Data\\TCS Historical Weekly.csv'
elif case == 3:
    url = 'E:\Lecs\IIIT\SMAI\Project\Data\\HCL Historical Weekly.csv'
elif case == 4:
    url = 'E:\Lecs\IIIT\SMAI\Project\Data\\Infy Historical Weekly.csv'
elif case == 5:
    url = 'E:\Lecs\IIIT\SMAI\Project\Data\\ONGC Historical Weekly.csv'
elif case == 6:
    url = 'E:\Lecs\IIIT\SMAI\Project\Data\\Reliance Historical Weekly.csv'

with open(url, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
data = numpy.array(data)
data = data[1:, 1:]
data = data.astype(float)
labels = ((data[:, 3] - data[:, 0]) > 0).astype(int)
data, labels = multiple_days_forward(data, 1)
print numpy.shape(labels)
print numpy.shape(data)


# In[157]:

def t_high(t, X):
    return max(X[:-t])


# In[158]:

def t_low(t, X):
    return min(X[:-t])


# In[159]:

def volume_high(t, X):
    return max(X[:-t])


# In[160]:

def volume_low(t, X):
    return min(X[:-t])


# In[161]:

def extract_features(data, indices):
    #remove the volume feature because of 0's
    data = data[:, [0, 1, 2, 3, 5]]
    #remove the first row because it is a header
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
    return features[:, indices], data


# In[162]:

features, data = extract_features(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
train_features = features[:500]
test_features = features[500:]
train_labels = labels[:500]
test_labels = labels[500:-1]


# In[163]:

clf = svm.SVC(kernel = 'rbf', C = 1.2, gamma = 0.001)
clf.fit(train_features, train_labels)


# In[164]:

predicted = clf.predict(test_features)
Accuracy = accuracy_score(test_labels, predicted)
Precision = recall_score(test_labels, predicted)
Recall = precision_score(test_labels, predicted)
print "Accuracy: ", Accuracy
print "Precision: ", Precision
print "Recall: ", Recall


# In[165]:

bought_price = list()
current_holdings = 0
sell_price = list()
for i in range(len(predicted)):
    if predicted[i]:
        current_holdings += 1
        bought_price.append(data[500+(i+1), 0])
    else:
        for j in range(current_holdings):
            sell_price.append(data[500+(i+1), 0])
        current_holdings = 0
print sum(sell_price) - sum(bought_price)


# In[166]:

step = numpy.arange(0, len(test_labels))
plt.subplot(211)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.ylabel('Actual Values')
plt.plot(step, test_labels, drawstyle = 'step')
plt.subplot(212)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.xlabel('Days')
plt.ylabel('Predicted Values')
plt.plot(step, predicted, drawstyle = 'step')
plt.show()
#plt.plot(plot_predicted)


# In[167]:

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


# In[168]:

trainer = BackpropTrainer(net, ds)


# In[169]:

epochs = 100
for i in range(epochs):
    trainer.train()


# In[170]:

predicted = list()
for i in test_features:
    #print net.activate(i)
    predicted.append(int(net.activate(i)>0.5))
predicted = numpy.array(predicted)


# In[171]:

print accuracy_score(test_labels, predicted)
print recall_score(test_labels, predicted)
print precision_score(test_labels, predicted)


# In[172]:

step = numpy.arange(0, len(test_labels))
plt.subplot(211)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.plot(step, test_labels, drawstyle = 'step')
plt.ylabel('Actual Values')
plt.subplot(212)
plt.xlim(-1, len(test_labels) + 1)
plt.ylim(-1, 2)
plt.plot(step, predicted, drawstyle = 'step')
plt.xlabel('Days')
plt.ylabel('Predicted Values')
plt.show()
#plt.plot(plot_predicted)


# In[173]:

bought_price = list()
current_holdings = 0
sell_price = list()
for i in range(len(predicted)):
    if predicted[i]:
        current_holdings += 1
        bought_price.append(data[500+(i+1), 0])
    else:
        for j in range(current_holdings):
            sell_price.append(data[500+(i+1), 0])
        current_holdings = 0
print sum(sell_price) - sum(bought_price)


# In[ ]:



