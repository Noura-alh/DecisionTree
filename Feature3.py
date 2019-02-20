import pandas as pd
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
import warnings
from graphviz import Source
import csv
from sklearn import tree

'''
This is with name destination and  name Origin encoded as integers 

'''



warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('/Users/Noura/Desktop/DT_algorthim/filtered_data.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
print(df.head())
print(df.isnull().values.any())

######## DATA  CLEANING ##############

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

randomState = 5
np.random.seed(randomState)

#X = X.loc[np.random.choice(X.index, 100000, replace = False)]

Y = X['isFraud']
del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis in the EDA
X = X.drop(['isFlaggedFraud'], axis = 1)

# Binary-encoding of labelled data in 'type'
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)

X.loc[X.nameOrig.str.contains('C'),'nameOrig'] = X.nameOrig.str.replace('C','')
X.loc[X.nameDest.str.contains('C'),'nameDest'] = X.nameDest.str.replace('C','')


X.nameOrig = X.nameOrig.astype(int)
X.nameDest = X.nameDest.astype(int)



nameDestList = []
nameOrginList = []
beforeDestSet = {}
beforeOrginSet = {}


for index, row in X.iterrows():
    #row['nameOrig'] = row['nameOrig'].replace("C", "")
    #row['nameDest'] = row['nameDest'].replace("C", "")
    nameDestList.append(row['nameDest'] )
    nameOrginList.append(row['nameOrig'])


beforeDestSet = set(nameDestList)

print('Number of unique Dest name BEFORE deleteing C',len(beforeDestSet))

beforeOrginSet = set(nameOrginList)
print('Number of unique Origin name BEFORE deleteing C',len(beforeOrginSet))


AfterNameDestList = []
AfterNameOrginList = []
uniqueNames = []
afterDestSet = {}
afterOrginSet = {}
numOfClients = {}

for index, row in X.iterrows():

    #AfterNameOrginList.append(row['nameOrig'])
    #AfterNameDestList.append(row['nameDest'])
    uniqueNames.append(row['nameOrig'])
    uniqueNames.append(row['nameDest'])


#nameOrig


for index, row in X.iterrows():
    #row['nameOrig'] = row['nameOrig'].replace("C", "")
    #row['nameDest'] = row['nameDest'].replace("C", "")
    AfterNameDestList.append(row['nameDest'] )
    AfterNameOrginList.append(row['nameOrig'])






afterDestSet = set(AfterNameDestList)

print('Number of unique Dest name AFTER deleteing C',len(afterDestSet))

afterOrginSet = set(AfterNameOrginList)
print('Number of unique Origin name AFTER deleteing C',len(afterOrginSet))

'''print('AFTER REMOVING C AfterNameOrginList:')
for each in AfterNameOrginList :
    print(each)

print('AFTER REMOVING C AfterNameDestList: ')
for each in AfterNameDestList :
    print(each)'''


Xfraud = X.loc[Y == 1]
XnonFraud = X.loc[Y == 0]

## ADDING FEATURES :
#X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
#X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
#X['is_Origin_duplicated'] = X.duplicated(['nameOrig'],keep= False)
#X['is_Dest_duplicated'] = X.duplicated(['nameDest'],keep= False)
X['AvgAmountOfTransaction'] = 0


for name in afterDestSet:
   X.loc[ X['nameDest'] == name, 'AvgAmountOfTransaction']=X.loc[X.nameDest == name].amount.mean()








numOfClients = set(uniqueNames)

print('Number of unique Clients in the dataset: ',len(numOfClients))










print('skew = {}'.format( len(Xfraud) / float(len(X)) ))
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, \
                                                random_state = randomState)
# Long computation in this cell (~1.8 minutes)
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())


clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, \
                n_jobs = 4)
clf.fit(trainX, trainY)
predict=clf.predict(testX)

''''##uncomment to train
clf = tree.DecisionTreeClassifier()
clf.fit(trainX,trainY)
tree.export_graphviz(clf.tree_, out_file='treepic.dot', feature_names=X.columns)
predict=clf.predict(testX)'''






#print('AUPRC = {}'.format(average_precision_score(testY, \ probabilities[:, 1])))
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(clf, height=1, color=colours, grid=False, \
                     show_values=False, importance_type='cover', ax=ax);
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

ax.set_xlabel('importance score', size=16);
ax.set_ylabel('features', size=16);
ax.set_yticklabels(ax.get_yticklabels(), size=12);
ax.set_title('Ordering of features by importance to the model learnt', size=20);
plt.show()




print(dict(zip(X.columns, clf.feature_importances_)))

cm = confusion_matrix(testY,predict)
print(cm)


plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

'''plotting decision tree'''

'''from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=X.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())'''




X.to_csv('Original.csv', encoding='utf-8', index=False)
