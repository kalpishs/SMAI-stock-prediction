import urllib2 
import csv
import numpy as np
from sklearn import svm 
from sklearn.metrics import *
fp1 = open("dataset/infy_15_4.csv", 'a+')
fp2= open("dataset/infy_15_4.csv",'r+')

all_features = []
timestamp_list =[]
close_list = []
high_list = []
low_list = []
open_price_list =[]
volume_list = []
open_change_percentage_list=[]
close_change_percentage_list=[]
low_change_percentage_list=[]
high_change_percentage_list=[]
volume_change_percentage_list=[]
def read_data(passing_for_url,fp):
    count=0
    if passing_for_url==1:
        datasetname= urllib2.urlopen('http://chartapi.finance.yahoo.com/instrument/1.0/HCLTECH.NS/chartdata;type=quote;range=1d/csv')
    else:
        datasetname = fp
    for line in datasetname:
        #count+=1
        l=line.split(',')
        #print l
        if(passing_for_url==1):
            if count > 17:
                fp.write(line)
            else:
                count+=1
                continue
        x = list(l[len(l)-1])
        x = x[0:len(x)-1]
        x = ''.join(x)
        l[len(l)-1]=x
        #print l
        all_features.append(l)
        timestamp, close, high, low, open_price , volume = l
        timestamp_list.append(int(timestamp))
        close_list.append(float(close))
        high_list.append(float(high))
        low_list.append(float(low))
        open_price_list.append(float(open_price))
        volume_list.append(float(volume))

                          
           

choice=int(input("chose URL(1) or file(2) :"))
if choice==1:
    read_data(choice,fp1)
    fp1.close()
else:
    read_data(choice,fp2)
    fp2.close()



for i in range(1, len(close_list)-1):
    close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
    close_change_percentage_list.append(close_change_percentage)
    
    open_change_percentage = (open_price_list[i+1] - open_price_list[i])/open_price_list[i]
    open_change_percentage_list.append(open_change_percentage)

    high_change_percentage = (high_list[i] - high_list[i-1])/high_list[i-1]
    high_change_percentage_list.append(high_change_percentage)
    if volume_list[i-1]==0:
        volume_list[i-1] = volume_list[i-2]

    volume_change_percentage = (volume_list[i] - volume_list[i-1])/volume_list[i-1]
    volume_change_percentage_list.append(volume_change_percentage)

    low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
    low_change_percentage_list.append(low_change_percentage)


close_change_percentage_list = np.array(close_change_percentage_list)
high_change_percentage_list = np.array(high_change_percentage_list)
low_change_percentage_list = np.array(low_change_percentage_list)
volume_change_percentage_list = np.array(volume_change_percentage_list)
open_price_list = np.array(open_price_list)
close_list = np.array(close_list)


label_list = close_list - open_price_list
label_list = label_list[1:-1]

for i in range(len(label_list)):
    if(label_list[i]>0):
        label_list[i]=1
    else:
        label_list[i]=0

feature = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list))
print feature


length_feature = len(feature)
len_train = int(0.75*length_feature)
train_feature = feature[0: len_train]
test_feature = feature[len_train: ]

train_label = label_list[0:len_train]
test_label = label_list[len_train:]





clf = svm.SVC(C=100000,kernel='rbf')
clf.fit(train_feature, train_label)

print clf.predict(test_feature)
print test_label
print accuracy_score(clf.predict(test_feature), test_label)



# import urllib2 
# import csv
# import numpy as np
# from sklearn import svm 
# from sklearn.metrics import *
# datasetname= urllib2.urlopen('http://chartapi.finance.yahoo.com/instrument/1.0/INFY.NS/chartdata;type=quote;range=1d/csv')
# count=0

# fp = open("dataset/infy_17_3.csv", 'w')

# all_features = []
# timestamp_list =[]
# close_list = []
# high_list = []
# low_list = []
# open_price_list =[]
# volume_list = []
# open_change_percentage_list=[]
# close_change_percentage_list=[]
# low_change_percentage_list=[]
# high_change_percentage_list=[]
# volume_change_percentage_list=[]
# for line in datasetname: # files are iterable
#     count+=1
#     if count >17:
#     	l=line.split(',')
#     	fp.write(line)
#     	x = list(l[len(l)-1])
#     	x = x[0:len(x)-1]
#     	x = ''.join(x)
#     	l[len(l)-1]=x
#     	all_features.append(l)
#     	timestamp, close, high, low, open_price , volume = l
#     	timestamp_list.append(int(timestamp))
#     	close_list.append(float(close))
#     	high_list.append(float(high))
#     	low_list.append(float(low))
#     	open_price_list.append(float(open_price))
#     	volume_list.append(float(volume))

# fp.close()
# for i in range(1, len(close_list)-1):
# 	close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
# 	close_change_percentage_list.append(close_change_percentage)
	
# 	open_change_percentage = (open_price_list[i+1] - open_price_list[i])/open_price_list[i]
# 	open_change_percentage_list.append(open_change_percentage)

# 	high_change_percentage = (high_list[i] - high_list[i-1])/high_list[i-1]
# 	high_change_percentage_list.append(high_change_percentage)

# 	volume_change_percentage = (volume_list[i] - volume_list[i-1])/volume_list[i-1]
# 	volume_change_percentage_list.append(volume_change_percentage)

# 	low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
# 	low_change_percentage_list.append(low_change_percentage)


# close_change_percentage_list = np.array(close_change_percentage_list)
# high_change_percentage_list = np.array(high_change_percentage_list)
# low_change_percentage_list = np.array(low_change_percentage_list)
# volume_change_percentage_list = np.array(volume_change_percentage_list)
# open_price_list = np.array(open_price_list)
# close_list = np.array(close_list)


# label_list = close_list - open_price_list
# label_list = label_list[1:-1]

# for i in range(len(label_list)):
# 	if(label_list[i]>0):
# 		label_list[i]=1
# 	else:
# 		label_list[i]=0

# feature = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list))
# print feature

# clf = svm.SVC()
# clf.fit(feature, label_list)

# print clf.predict(feature)
# print accuracy_score(clf.predict(feature), label_list)


