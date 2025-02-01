#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from collections import Counter
from IPython.display import Image  


# Configuring the notebook
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading the data
credit_card = pd.read_csv('F:/MS/UMich-Dearborn/Sem1/CIS 5570/Project/creditcard.csv')
credit_card.head()
     


# In[4]:


#Data Analysis
pd.set_option("display.float", "{:.2f}".format)

credit_card.describe()


# In[5]:


#Check for null values or missing values 
credit_card.isnull().sum().sum()


# In[6]:


#The data consists of 284807 rows and 31 columns. We checked for missing values, there are none. 'Time' and 'Amount' are the only 
#features that have not been transformed. Here, 'Time' contains seconds elapsed between each transaction and he first transaction 
#of the dataset. 'Amount' is the transaction Amount.
credit_card[['Time', 'Amount']].describe()


# In[12]:


#Histogram for 'Time' and 'Amount' features considering the Class column, which represents the frauds.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,8))
sns.histplot(credit_card['Time'][credit_card['Class'] == 1], bins=15, ax=ax1)
sns.histplot(credit_card['Time'][credit_card['Class'] == 0], bins=15, ax=ax2)

sns.histplot(credit_card['Amount'][credit_card['Class'] == 1], bins=5, ax=ax3)
sns.histplot(credit_card['Amount'][credit_card['Class'] == 0], bins=5, ax=ax4)

ax1.set_title('Fraud')
ax2.set_title('Non Fraud')
ax3.set_title('Fraud')
ax4.set_title('Non Fraud')
plt.tight_layout()
plt.show()


# In[17]:


LABELS = ["Normal", "Fraud"]

count_classes = pd.value_counts(credit_card['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title('Transaction Class Distribution')
plt.xticks(range(2), LABELS)
plt.xlabel('Class')
plt.ylabel('Frequency');


# In[18]:


credit_card.Class.value_counts()


# In[21]:


#We are able to see how imbalanced the data is the dataset. Most of the transactions are non-fraud. 
#If we use this DataFrame as the base for our predictive models and analysis, we might get a lot of errors, 
#and  algorithms might overfit since they will “assume” that most transactions are not a fraud. 
#we want model to detect patterns that give signs of fraud.
fraud = credit_card[credit_card['Class']==1]
normal = credit_card[credit_card['Class']==0]

print(f"Shape of Fraudulant transactions: {fraud.shape}")
print(f"Shape of Non-Fraudulant transactions: {normal.shape}")


# In[25]:


#We want to know if there are features that influence heavily whether a specific transaction is a fraud.
#However, it is important that we use the correct DataFrame (subsample) in order for us to see which features have a
#high positive or negative correlation with regard to fraudulent transactions. So, we use correlation matrix.

corr = credit_card.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, cmap='RdBu')

plt.tight_layout()
plt.show()
     


# In[49]:


print(credit_card['Class'].value_counts(normalize=True))

fig, ax = plt.subplots(figsize=(8,4))
sns.countplot(credit_card['Class'])

plt.tight_layout()
plt.show()


# In[28]:


#Data Pre-processing
scaler = StandardScaler()

credit_card['std_amount'] = scaler.fit_transform(credit_card['Amount'].values.reshape(-1, 1))
credit_card['std_time'] = scaler.fit_transform(credit_card['Time'].values.reshape(-1, 1))

credit_card.drop(['Amount', 'Time'], axis=1, inplace=True)

credit_card.head()


# In[35]:


X = credit_card.drop('Class', axis=1)
y = credit_card['Class']
scalar = StandardScaler()
X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v, 
                                                            test_size=0.2, random_state=42)

X_train = scalar.fit_transform(X_train)
X_validate = scalar.transform(X_validate)
X_test = scalar.transform(X_test)

w_p = y_train.value_counts()[0] / len(y_train)
w_n = y_train.value_counts()[1] / len(y_train)

print(f"Fraudulant transaction weight: {w_n}")
print(f"Non-Fraudulant transaction weight: {w_p}")


# In[36]:


print(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}\n")
print(f"VALIDATION: X_validate: {X_validate.shape}, y_validate: {y_validate.shape}\n")
print(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")


# In[41]:


#RandomUnderSampler class to balance X_train and y_train
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_train, y_train)

print(pd.Series(y_rus).value_counts(normalize=True))

fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(y_rus)

plt.tight_layout()
plt.show()


# In[42]:


#the RandomOverSampler class to balance the data. We'll train models with both balancement methods and compare the results.
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X_train, y_train)

print(pd.Series(y_ros).value_counts(normalize=True))

fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(y_ros)

plt.tight_layout()
plt.show()
     


# In[53]:


#Models
#Decision Tree
#Setting the maximum depth of the tree is important to avoid problems like overfitting. 
#In order to find the best possible depth for the tree, we'll create models with a range of different depths and 
#then see which one provided the best result. We'll then use this best depth to create and evaluate the final 
#decision tree model.
n = 11
acc_tree = np.zeros((n-3))

for i in range(3, n):

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

    tree.fit(X_rus, y_rus)

    y_pred_tree = tree.predict(X_test)

    acc_tree[i-3] = accuracy_score(y_test, y_pred_tree)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(range(3, n), acc_tree, linewidth=3, marker='o')
ax.set_title('Accuracy Score by Tree Depth')
ax.set_ylabel('Accuracy Score')
ax.set_xlabel('Tree Depth')
ax.grid(False)

plt.tight_layout()
plt.show()

best_depth = acc_tree.argmax()+3
print(f'The best accuracy was {round(acc_tree.max(), 4)} with depth={best_depth}.') 


# In[56]:


#training model with underbalanced data:
tree_under = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)

tree_under.fit(X_rus, y_rus)

y_pred_tree_under = tree_under.predict(X_test)

def report(pred):
    print(classification_report(y_test, pred))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred, normalize='true'), annot=True, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.show()

    print(f'ROC AUC Score: {round(roc_auc_score(y_test, pred), 4)}')


report(y_pred_tree_under)


# In[69]:


#Visualisation
import pydotplus
dot = export_graphviz(tree_under, filled=True, rounded=True, feature_names=X.columns, class_names=['0', '1'])

graph = pydotplus.graph_from_dot_data(dot)  
Image(graph.create_png())


# In[70]:


tree_over = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)

tree_over.fit(X_ros, y_ros)

y_pred_tree_over = tree_over.predict(X_test)

report(y_pred_tree_over)


# In[71]:


dot = export_graphviz(tree_over, filled=True, rounded=True, feature_names=X.columns, class_names=['0', '1'])

graph = pydotplus.graph_from_dot_data(dot)  
Image(graph.create_png())


# In[75]:


#KNN model
Ks = 11
acc_knn = np.zeros((Ks-1))

for k in range(1, Ks):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_rus, y_rus)

    y_pred_knn = knn.predict(X_test)

    acc_knn[k-1] = accuracy_score(y_test, y_pred_knn)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(range(1, Ks), acc_knn, linewidth=3, marker='o')
ax.set_title('Accuracy Score by Number of Neighbors')
ax.set_ylabel('Accuracy Score')
ax.set_xlabel('Number of Neighbors')
ax.grid(False)

plt.tight_layout()
plt.show()

best_k = acc_knn.argmax()+1
print(f'The best accuracy was {round(acc_knn.max(), 4)} with k={best_k}.') 


# In[78]:


#Training the model with the underbalanced data
knn_under = KNeighborsClassifier(n_neighbors=best_k)

knn_under.fit(X_rus, y_rus)

y_pred_knn_under = knn_under.predict(X_test)

def report1(pred):
    print(classification_report(y_test, pred))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred, normalize='true'), annot=True, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.show()

    print(f'ROC AUC Score: {round(roc_auc_score(y_test, pred), 4)}')

report1(y_pred_knn_under)


# In[80]:


#training the model with the overbalanced data
knn_over = KNeighborsClassifier(n_neighbors=best_k)

knn_over.fit(X_ros, y_ros)

y_pred_knn_over = knn_over.predict(X_test)

report1(y_pred_knn_over)


# In[82]:


summary = pd.DataFrame(data={
'labels': ['Accuracy', 'Precision', 'Recall', 'F1_score', 'roc_auc'],
'decision_trees_under': [accuracy_score(y_test, y_pred_tree_under), precision_score(y_test, y_pred_tree_under), recall_score(y_test, y_pred_tree_under), f1_score(y_test, y_pred_tree_under), roc_auc_score(y_test, y_pred_tree_under)],
'decision_trees_over': [accuracy_score(y_test, y_pred_tree_over), precision_score(y_test, y_pred_tree_over), recall_score(y_test, y_pred_tree_over), f1_score(y_test, y_pred_tree_over), roc_auc_score(y_test, y_pred_tree_over)],
'knn_under': [accuracy_score(y_test, y_pred_knn_under), precision_score(y_test, y_pred_knn_under), recall_score(y_test, y_pred_knn_under), f1_score(y_test, y_pred_knn_under), roc_auc_score(y_test, y_pred_knn_under)],
'knn_over': [accuracy_score(y_test, y_pred_knn_over), precision_score(y_test, y_pred_knn_over), recall_score(y_test, y_pred_knn_over), f1_score(y_test, y_pred_knn_over), roc_auc_score(y_test, y_pred_knn_over)]
}).set_index('labels')
summary.index.name = None

summary


# In[83]:


fig, ax = plt.subplots(figsize=(12, 6))
summary.plot.bar(ax=ax)
ax.legend(bbox_to_anchor=(1, 1), frameon=False)
ax.grid(False)
ax.set_title('Models Comparison for Each Metric')

plt.xticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:




