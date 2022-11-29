# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.sparse as sp
from GAE.train_model import gae_model
import os
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#hyperparameter
avaerage_num = 50
dim = 40

#load data
def merge_matrix():
    top = pd.merge(LncLnc,Lnc_type,how='left',left_index=True,right_index=True)
    up = pd.merge(np.transpose(Lnc_type),CanCan,how='left',left_index=True,right_index=True)
    result = pd.concat([top,up],axis=0)
    return result

LncRNACancer = pd.read_excel('LncCancer.xlsx',engine='openpyxl',index_col=0)
Lnc_type = pd.read_excel('Lnc_type.xlsx',engine='openpyxl',index_col=0)
CanCan = pd.DataFrame(index=list(LncRNACancer.columns),columns=list(LncRNACancer.columns),dtype=object)
CanCan = CanCan.fillna(0)
adj = merge_matrix()

#LncRNA、cancer_event list
LncList = list(LncRNACancer.index)
EventList = list(LncRNACancer.columns)

#VGAE parameter
class Args():
    def __init__(self):
        self.lr = 0.01
        self.epochs = 200
        self.hidden = 200
        self.dimensions = dim
        self.weight_decay = 5e-4
        self.dropout = 0.1
        self.gae_model_selection = 'gcn_vae'        


adj_csr = sp.csr.csr_matrix(adj)
arg = Args()

for i in range(avaerage_num):
    gae = gae_model(arg)
    gae.train(adj_csr)
    # save latent feature
    gae.save_embeddings(str(i) + '.txt', list(adj.index))  # Modify Path
    print(str(i) + '.txt saved!')

feature = {}
# create a dic
for k in list(adj.index):
    feature[k] = [0] * dim

for i in range(avaerage_num):
    with open(str(i) + '.txt', 'r') as f:  # Consistent with  70 line path
        data = f.read().split('\n')[1:]

    data.pop()
    for line in data:
        node_feature = line.split('\t')
        feature[node_feature[0]] = [d[0] + float(d[1]) for d in zip(feature[node_feature[0]], node_feature[1:])]
    print(str(i) + '.txt done')

# Calculating the average feature
for key in feature.keys():
    feature[key] = [i / avaerage_num for i in feature[key]]
print('Average feature calculation completed!')

# save node feature
with open('reslult_event_{}.txt'.format(dim), 'w') as f:  # Modify path
    for key, values in feature.items():
        f.write(key + '\t')
        count = 1
        for num in values:
            if count == len(values):
                f.write(str(num) + '\n')
            else:
                f.write(str(num) + '\t')
                count += 1
print('average feature saved')

# Delete temporary files
command = ''
for i in range(avaerage_num):
    command += os.path.join('path', str(i) + '.txt ')  # Consistent with  70 line path
os.system('del ' + command)

# Process node feature
with open('reslult_event_{}.txt'.format(dim), 'r') as f:  # Consistent with  94 line path
    feature = f.read().split('\n')
feature.pop(-1)
count, lnc_node, event_node = 0, {}, {}
for line in feature:
    count += 1
    node_feature = line.split('\t')
    if count <= 228:
        lnc_node[node_feature[0]] = list(node_feature[1:])
    else:
        event_node[node_feature[0]] = list(node_feature[1:])

# Dividing positive and negative dataset
pos_set = []
neg_set = []
for col in range(LncRNACancer.shape[0]):
    for ind in range(LncRNACancer.shape[1]):
        if LncRNACancer.iloc[col, ind] == 0:
            neg_set.append([col, ind, 0])
        else:
            pos_set.append([col, ind, 1])
LncList = list(LncRNACancer.index)
EventList = list(LncRNACancer.columns)


#5-CV TO ROC,predict
result_dic = {}
kf = KFold(n_splits=5,random_state=1412,shuffle=True)  
tprs,aucs,mean_fpr = [],[],np.linspace(0,1,100) 
recall_combind,auprs,mean_precision = [],[],np.linspace(0,1,100)
count = 0
color = ['green','yellow','red','violet','tomato']
test = neg_set[len(pos_set):]
fig,ax = plt.subplots()
for train_index,test_index in kf.split(pos_set):
    train_set = []
    for index in train_index:
        train_set.append(pos_set[index])
    for i in range(len(train_index)):
        train_set.append(neg_set[i])
    x_train,y_train = [],[]
    for k in train_set:
        x_train.append(lnc_node[LncList[k[0]]]+cancer_node[cancerList[k[1]]])
        y_train.append(k[2])
    test_set = []
    for index in test_index:
        test_set.append(pos_set[index])
    test_set.extend(neg_set[len(test_index):len(pos_set)])
    x_test,y_test,x_pred = [],[],[]
    for k in test_set:
        x_test.append(lnc_node[LncList[k[0]]]+cancer_node[cancerList[k[1]]])
        y_test.append(k[2])
    for k in test:
        x_pred.append(lnc_node[LncList[k[0]]]+cancer_node[cancerList[k[1]]])
    #train_model
    model = KNeighborsClassifier(n_neighbors=9)
    model = model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    ass_pred = model.predict(x_pred)
    result_dic[count] = ass_pred
    score = model.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test,score)
    tprs.append(np.interp(mean_fpr,fpr,tpr))
    tprs[-1][0]= 0.0
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    ax.plot(fpr,tpr,lw=1,alpha=.8,label='ROC fold {0} (AUC={1:.2f})'.format(count+1,roc_auc),color=color[count])
    count += 1


ax.plot([0,1],[0,1],linestyle='--',lw=2,color='r',alpha=.8)
mean_tpr = np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc = auc(mean_fpr,mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (AUC = %0.2f)'\
         % (mean_auc),lw=2)
ax.set_xlabel('False Positive Rate',fontweight='bold',fontsize=20,fontproperties='Times New Roman',weight='bold')
ax.set_ylabel('True Positive Rate',fontweight='bold',fontsize=20,fontproperties='Times New Roman',weight='bold')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.setp(ax.legend().texts,family='Times New Roman')
std_tpr=np.std(tprs,axis=0)
#tprs_upper=np.minimum(mean_tpr+std_tpr,1)
#tprs_lower=np.maximum(mean_tpr-std_tpr,0)
#plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
ax.set_xlim([-0.05,1.05])
ax.set_ylim([-0.05,1.05])
plt.savefig('lnc_cancertype_roc.pdf',format="pdf",dpi=600)


#PR
kfold = KFold(n_splits=5,random_state=1412,shuffle=True)
y_real,y_proba,count = [],[],0
fig1,ax1 = plt.subplots()
for train_index,test_index in kfold.split(pos_set):
    train_set = []
    for index in train_index:
        train_set.append(pos_set[index])
    for i in range(len(train_index)):
        train_set.append(neg_set[i])
    x_train,y_train = [],[]
    for k in train_set:
        x_train.append(lnc_node[LncList[k[0]]]+cancer_node[cancerList[k[1]]])
        y_train.append(k[2])
    test_set = []
    for index in test_index:
        test_set.append(pos_set[index])
    test_set.extend(neg_set[len(test_index):len(pos_set)])
    x_test,y_test = [],[]
    for k in test_set:
        x_test.append(lnc_node[LncList[k[0]]]+cancer_node[cancerList[k[1]]])
        y_test.append(k[2])
    model = KNeighborsClassifier(n_neighbors=9)
    model = model.fit(x_train,y_train)
    pred_proba = model.predict_proba(x_test)[:,1]
    precison, recall, _ =precision_recall_curve(y_test,pred_proba)
    lab = 'PR fold %d (AUPR=%.2f)'%(count+1,auc(recall,precison))
    ax1.plot(recall,precison,label=lab,alpha=.8,color=color[count])
    y_real.append(y_test)
    y_proba.append(pred_proba)
    count += 1

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real,y_proba)
ax1.plot([0,1],[1,0],linestyle='--',lw=2,color='r',alpha=.8)
ax1.plot(recall,precision,label='Mean PR (AUPR={0:.2f})'.format(auc(recall,precision)),\
         lw=2)
ax1.set_xlabel('Recall',fontweight='bold',fontsize=20,fontproperties='Times New Roman',weight='bold')
ax1.set_ylabel('Precision',fontweight='bold',fontsize=15,fontproperties='Times New Roman',weight='bold')
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.setp(ax1.legend().texts,family='Times New Roman')
fig1.savefig('lnc_cancertype_pr.pdf',dpi=600,format="pdf")

res,pred_list = [0]*len(result_dic[0]),[]
for i in range(len(result_dic)):
    res += np.array(result_dic[i])

for i,j in zip(res,test):
    if i >= 5:    #threshold
        pred_list.append([LncList[j[0]],cancerList[j[1]]])

with open('result.txt','w') as f:
    for i in pred_list:
        f.write(i[0]+'\t')
        f.write(i[1]+'\n')

res_dic = {}
for i in pred_list:
    if i[0] not in res_dic.keys():
        res_dic[i[0]] = 1
    elif i[0] in res_dic.keys():
        res_dic[i[0]] += 1
