import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
l1=np.zeros([108])
l2=np.ones([48])
l3=2*np.ones([55])
l4=3*np.ones([51])
l5=4*np.ones([25])
true=np.concatenate((l1,l2,l3,l4,l5),axis=0)
x=np.loadtxt('results/all_region_pre_label50.txt',delimiter=",",dtype=np.str)
y=np.array(x)
y = y.astype(np.float)
row,l=y.shape
ma=np.zeros([5,5])
for j in[4]:
    for i in range(row):
        pre=y[i,:]
        tre=true
        pre[pre!=j]=5
        tre[tre!=j]=5
        cm = confusion_matrix(pre, true)
        print(cm)
        tpr=cm[0,0]/(cm[0,0]+cm[1,0])
        fpr = cm[0,1] / (cm[1,1] + cm[0, 1])
        results_txt=str(tpr) + '\t'+str(fpr)+'\n'
        with open('./results/' + 'roc' + str(j) + '.txt', "a+") as f:
            f.write(results_txt)




