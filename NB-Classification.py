
# coding: utf-8

# # SMS Spam Classification using Multinomial Naive Bayes
# 
# In this project, we will create two models based on Multinomial NB. 
# First model will classify using the standard formula for Multinomial Naive Bayes.
# The second model will take in account the length of documents as fefatures, since length have a visible effect on class (Spam or Ham) as shown in the graph you will see shortly.
# 
# ```P(cl|doc,len) = (P(doc,len|cl) * P(cl)) / P(doc,len)
#               = (P(doc|cl) * P(len|cl) * P(cl)) / (P(doc) * P(len))
#               = (P(doc|cl) * P(cl)) / P(doc) * P(len|cl) / P(len)
#               = P(cl|doc) * P(len|cl) / P(len)```

# In[212]:

# Importing Necessary Modules

import pandas as pd
import string
from nltk.corpus import stopwords
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ### Reading Data
# 
# We will read the raw text file and store the data in an organized way into a csv. We will create a column SPAM. Inside the SPAM column, 1 denotes that the document is spam, 0 denotes it is not spam.

# In[213]:

data_file = 'SMSSpamCollection.txt'

with open(data_file,'r') as f:
    csv_file = open("data.csv", "w+") 
    columnTitleRow = "SMS,SPAM\n"
    csv_file.write(columnTitleRow)
    counter = 0
    spamCounter = 0
    hamCounter = 0
    for line in f:
        counter+=1
        if counter%500==0:
            print(counter," Rows Written")
        if 'spam' in line[:4]:
            line = line[4:]
            t=1
            spamCounter+=1
        else:
            line = line[3:]
            t=0
            hamCounter+=1
        row = line.replace("\n","").replace(",","").replace("\t","") + "," + str(t) + "\n"
        csv_file.write(row)
csv_file.close()

print("Total Spam SMS: ",spamCounter)
print("Total Ham SMS: ",hamCounter)


# ### Loading Dataset 
# 
# Now let's load data into dataframe from the csv we created above

# In[214]:

df = pd.read_csv('data.csv')
print(df.head())


# ##### Declaring some variables that will be used in calculating the probabilities

# In[215]:

data_info = df.groupby('SPAM').size()
TOTAL_SPAM = data_info[1]
TOTAL_HAM = data_info[0]
PRIOR_SPAM = TOTAL_SPAM/(TOTAL_SPAM + TOTAL_HAM)
PRIOR_HAM = TOTAL_HAM/(TOTAL_SPAM + TOTAL_HAM)


# ### Extracting New Feature
# 
# Creating a new feature LENGTH, which will be used in another model to demonstrate the improvement in accuracy.

# In[216]:

df['LENGTH'] = df['SMS'].apply(len)
print(df.head())
print("\n")
print("Max length of HAM: ",training_df[(training_df['SPAM']==0)].LENGTH.max())
print("Max length of SPAM: ",training_df[(training_df['SPAM']==1)].LENGTH.max())


# It can be clearly seen that for spam sms, the length is not more than 200. For Hams, the legth goes upto 910

# In[217]:

sns.lmplot( x="SPAM", y="LENGTH", data=df, fit_reg=False, hue='SPAM', legend=False)
sns.plt.show()


# ### Removing Punctuation and building Vocab

# In[218]:

## Removing punctuation
t = [''.join(c for c in s if c not in string.punctuation) for s in df['SMS'].values.flatten()]
# Building vocab and removing stop words
vocab = collections.Counter([y for x in t for y in x.split(" ") if y.lower() not in stopwords.words('english')])


# In[219]:

# Most Frequent terms and their frequency
vocab.most_common(10)


# In[220]:

V = len(vocab)


# ### Splitting data into Training and Validation sets.

# In[221]:

validation_df = df.iloc[:500]
training_df = df.iloc[500:]


# In[222]:

# Calculating terms in spam and ham

terms_in_spam = terms_in_ham = 0
for index, row in training_df.iterrows():
    if row['SPAM']==1:
        terms_in_spam += len(row['SMS'].split(" "))
    elif row['SPAM']==0:
        terms_in_ham += len(row['SMS'].split(" "))
print(terms_in_spam, terms_in_ham)


# # BASIC MULTINOMIAL NB CLASSIFIER
# 
# We were able to achieve accuracy of 87.8% using the standard Multiomial NB classification.

# In[223]:

def single_predict(sms):
    sms = [''.join(c for c in s if c not in string.punctuation) for s in sms]
    sms = ''.join(sms).split(" ")
    spam_prob = PRIOR_SPAM
    ham_prob = PRIOR_HAM
    terms_in_spam = terms_in_ham = 0
    flag = 0
    for term in sms:
        term = " "+term+" "
        spam_count = ham_count = 0
        for index, row in training_df[(training_df['SMS'].str.contains(term))].iterrows():
            
            if row['SPAM']==1:
                spam_count+=1
                
            else:
                ham_count+=1

    spam_prob = spam_prob * ( (spam_count + 1) / (terms_in_spam + V) )
    ham_prob = ham_prob * ( (ham_count + 1) / (terms_in_ham + V) )
                
    if spam_prob>ham_prob:
        return 1
    else:
        return 0


# In[224]:

def Standard_NB_Classifier(df):
    total = 0
    correct = 0
    correct_spam = 0
    correct_ham = 0
    for index, row in df.iterrows():
        if single_predict(row['SMS'])==row['SPAM']:
            correct+=1
            if row['SPAM'] == 1:
                correct_spam = correct_spam+1
            else:
                correct_ham = correct_ham+1
        total+=1
        if(total%50==0):
            print("Querries Processed: ",total)
            
    print("Accuracy: ",(correct/total)*100,"%")
    print("Correct Predictions: ",correct,"/",len(df.index))
            
    data_info = df.groupby('SPAM').size()
    TOTAL_SPAM = data_info[1]
    TOTAL_HAM = data_info[0]    
    
    print("Confusion Matrix")
    print("[",correct_spam,TOTAL_SPAM,"]\n[",correct_ham,TOTAL_HAM,"]")
    


# In[225]:

Standard_NB_Classifier(validation_df)


# ### MULTINOMIAL NB USING LENGTH AS A FEATURE
# 
# We were able to increase the accuracy upto 89% by using length of documents as a feature.

# In[226]:

def single_predict_improved(sms):
    sms = [''.join(c for c in s if c not in string.punctuation) for s in sms]
    sms = ''.join(sms).split(" ")
    avg_length_spam = 0
    avg_length_ham = 0
    spam_prob = PRIOR_SPAM
    ham_prob = PRIOR_HAM
    terms_in_spam = terms_in_ham = 0
    flag = 0
    for term in sms:
        term = " "+term+" "
        spam_count = ham_count = 0
        for index, row in training_df[(training_df['SMS'].str.contains(term))].iterrows():
            if row['SPAM']==1:
                spam_count+=1
                avg_length_spam = avg_length_spam + row['LENGTH']
            else:
                ham_count+=1
                avg_length_ham = avg_length_ham + row['LENGTH']
                
    avg_length_spam = avg_length_spam / len(training_df[(training_df['SPAM']==1)].index)
    avg_length_ham = avg_length_ham / len(training_df[(training_df['SPAM']==0)].index)
    spam_prob = (spam_prob * ( (spam_count + 1) / (terms_in_spam + V) )) * ((avg_length_spam + 1) / (len(sms) + 1))
    ham_prob = (ham_prob * ( (ham_count + 1) / (terms_in_ham + V) )) * ((avg_length_ham + 1) / (len(sms) + 1))
             

    if spam_prob>ham_prob:
        return 1
    else:
        return 0


# In[227]:

def Improved_NB_Classifier(df):
    total = 0
    correct = 0
    correct_spam = 0
    correct_ham = 0
    for index, row in df.iterrows():
        if single_predict_improved(row['SMS'])==row['SPAM']:
            correct+=1
            if row['SPAM'] == 1:
                correct_spam = correct_spam+1
            else:
                correct_ham = correct_ham+1
        total+=1
        if(total%50==0):
            print("Querries Processed: ",total)
            
    print("Accuracy: ",(correct/total)*100,"%")
    print("Correct Predictions: ",correct,"/",len(df.index))
            
    data_info = df.groupby('SPAM').size()
    TOTAL_SPAM = data_info[1]
    TOTAL_HAM = data_info[0]    
    
    print("Confusion Matrix")
    print("[",correct_spam,TOTAL_SPAM,"]\n[",correct_ham,TOTAL_HAM,"]")


# In[228]:

Improved_NB_Classifier(validation_df)


# In[ ]:



