import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
a = np.loadtxt("Infy Historical Daily.csv",usecols=(0,-1),dtype=str,skiprows=1,delimiter=',')
adates = a[:,0]
aclose = a[:,1].astype(float)
del a
adict = {}
for i in range(adates.shape[0]):
    adict[datetime.strptime(adates[i], "%Y-%m-%d")]=aclose[i]
    #adict[adates[i]]=aclose[i]
del adates
del aclose
b= np.loadtxt("INFOSYS share.csv",dtype=str,delimiter=',')
bdates=b[:,0]
bsearches = b[:,1].astype(float)
del b
bdict = {}
for i in range(bdates.shape[0]):
    bdict[(datetime.strptime(bdates[i].split(' - ')[1],"%Y-%m-%d"))-(timedelta(days=1))]=bsearches[i]
    #bdict[bdates[i].split(' - ')[1]]=bsearches[i]
l=[]
for i in adict.keys():
    if i in bdict.keys():
        l.append([i,adict[i],bdict[i]])

print len(l)
l = sorted(l,key= lambda x: x[0])
j=l

for i in range(0,len(l)):
    j[i][1]=abs(l[i][1]-l[i-1][1])/5
j = j[1:]

x = [i[0] for i in j]
y1 = [i[1] for i in j]
y2 = [i[2] for i in j]
plt.plot(x,y1,label='stock trend:INFOSYS')
plt.plot(x,y2,label='Google trend:INFOSYS share')
plt.legend()
plt.show()
