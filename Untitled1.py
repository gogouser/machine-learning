
# coding: utf-8

# In[4]:


get_ipython().system('pip install --user matplotlib numpy scipy sklearn')


# In[7]:


get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import requests
from io import BytesIO

iris_url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
resp = requests.get(iris_url)
print(resp.content)

data = BytesIO(resp.content)



# In[9]:


import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv(data)
print(df.head())


# In[14]:


print(df.head())


# In[28]:


target = df[df.columns[-1]]
target = target.astype('category')
numeric_data = df._get_numeric_data()
print('')
print(target.head())
print('')
print(numeric_data.head())


# In[41]:


training_data, testing_data, training_lable, testing_lable = train_test_split(numeric_data,target.cat.codes)

print(training_data.head())
print()
print(len(training_data))
print(len(testing_data))




# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')


# In[43]:


matrix = confusion_matrix(testing_lable, predict_result)
report = classification_report(
    testing_label,
    predict_result,
    target_names=target.cat.categories
)
acc = accuracy_score(testing_lable,predict_result)

print(matrix)
print('====================')
print(report)
print('====================')
print(acc)

