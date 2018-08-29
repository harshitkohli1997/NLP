
# coding: utf-8

# In[1]:


import nltk
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))


# In[3]:


for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')


# In[7]:


import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep = '\t',names = ['label','message'])


# In[8]:


messages.describe()


# In[9]:


messages.groupby('label').describe()


# In[10]:


messages['length'] = messages['message'].apply(len)


# In[12]:


messages.head()


# In[13]:


import string


# In[14]:


mess = 'Sample message! notice: it has puntuation.'


# In[18]:


nopunc = [c for c in mess if c not in string.punctuation]


# In[21]:


nopunc = ''.join(nopunc)


# In[28]:


nopunc.split()
from nltk.corpus import stopwords
stopwords.words('english')[0:10]


# In[29]:


#clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[30]:


clean_mess


# In[31]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[32]:


messages['message'].head(5).apply(text_process)


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer


# In[36]:


bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])


# In[37]:


print(len(bow_transformer.vocabulary_))


# In[38]:


mess4 = messages['message'][3]


# In[39]:


print(mess4)

