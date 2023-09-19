import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
l1=np.zeros([108])
l2=np.ones([48])
l3=2*np.ones([55])
l4=3*np.ones([51])
l5=4*np.ones([25])
true=np.concatenate((l1,l2,l3,l4,l5),axis=0)
x=np.loadtxt('results/pre_labelAD_reho50.txt',delimiter=",",dtype=np.str)
y=np.array(x)
y = y.astype(np.float)
row,l=y.shape
ma=np.zeros([5,5])
for i in range(row):
    con=confusion_matrix(y[i,:],true)
    ma=ma+con
cm=ma/30

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




attack_types = ['HC', 'GAD', 'SP','PD','SAD']
classes=['HC', 'GAD', 'KG','PD','SAD']
plot_confusion_matrix(cm, classes=attack_types, normalize=True,)


def calculate_all_prediction(confMatrix):
    '''
    计算总精度,对角线上所有值除以总数
    :return:
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    print('准确率:' + str(prediction) + '%')

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    ka=(po - pe) / (1 - pe)
    print('kappa'+str(ka))


def calculae_lable_prediction(confMatrix):
    '''
    计算每一个类别的预测精度:该类被预测正确的数除以该类的总数
    '''
    l = len(confMatrix)
    pre=[]
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=1)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)

        pre.append(prediction)
    print('精确率:' +str(np.mean(pre))+ '%')

def calculate_label_recall(confMatrix):
    l = len(confMatrix)
    recall=[]
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=0)[i]
        label_correct_sum = confMatrix[i][i]
        re= round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        recall.append(re)

    print('召回率:' +str(np.mean(recall))+ '%')




calculate_all_prediction(cm)
calculae_lable_prediction(cm)
calculate_label_recall(cm)
kappa(cm)