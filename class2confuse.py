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
ma=np.zeros([2,2])
for j in[0]:
    for i in range(row):
        pre=y[i,:]
        tre=true
        pre[pre!=j]=5
        tre[tre!=j]=5
        cm = confusion_matrix(pre, true)
        ma=ma+cm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,x=0.5,y=-0.3,fontsize='xx-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,fontsize=13)
    plt.yticks(tick_marks, classes,fontsize=13)
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=10)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=14)
    plt.xlabel('Predicted label',labelpad=10,fontsize=14)
    plt.show()




attack_types = ['HC', 'AD']

plot_confusion_matrix(ma, classes=attack_types, normalize=True,)




