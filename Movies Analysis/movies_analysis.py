
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


def count_total_awards_from_string(str):
     return sum([int(s) for s in str.split() if s.isdigit()])

def str_to_int(str):
    return int(str)

def strNA_to_NAN(str):
    if (str == 'N/A') :
        return np.nan
    return str
        


# In[3]:


wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True,encoding= "utf-8")
rottenTomatos = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True,encoding= "utf-8")
omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True,encoding= "utf-8") 
genres =  pd.read_json('genres.json.gz', orient='record', lines=True,encoding= "utf-8") 


# In[4]:


# score critics corelation
# score award corealtion
# score made profit


# In[5]:


# score critics corelation
task1 = rottenTomatos.sort_values(by = ['audience_average'],ascending = False).dropna(subset =['critic_average','audience_average'])
x = np.array(task1.audience_average)
y = np.array(task1.critic_average)
reg = stats.linregress(x,y)
task1['fit'] = task1['audience_average']*reg.slope+ reg.intercept
plt.plot(task1['audience_average'], task1['critic_average'],'.')
plt.plot(task1['audience_average'], task1['fit'],'r-',linewidth = 2)
plt.ylabel('critic_average')
plt.xlabel('audience_average')
plt.title('ratings correaltion')
plt.savefig('ratings-correlation.png')
plt.show()

r_value, p_value = stats.pearsonr(x,y)
print("correlation coefficient of audience ratings and critics ratings:", r_value)


# In[6]:


# score/ award corealtion


# In[7]:


task2 = task1.join(omdb.set_index('imdb_id'), on='imdb_id')
task2.omdb_awards = task2.omdb_awards.apply(strNA_to_NAN)
task2 = task2.dropna(subset =['omdb_awards'])
task2['counts'] = task2['omdb_awards'].apply(count_total_awards_from_string)
task2['counts'] = task2['counts'].fillna(value = '0')


# In[8]:


task2.counts = task2.counts.apply(str_to_int)


# In[9]:


x = np.array(task2.audience_average)
y = np.array(task2.counts)
reg = stats.linregress(x,y)
task2['fit'] = task2['audience_average']*reg.slope+ reg.intercept
plt.plot(task2['audience_average'], task2['counts'],'.')
plt.plot(task2['audience_average'], task2['fit'],'r-',linewidth = 2)
plt.ylabel('Award counts')
plt.xlabel('audience_average')
plt.title('ratings/award correaltion')
plt.savefig('ratings-award-correlation.png')
plt.show()
r_value, p_value = stats.pearsonr(x,y)
print("correlation coefficient of audience ratings and awards won:", r_value)


# In[10]:


# score genre corelation
task3 = rottenTomatos.join(wikidata.set_index('rotten_tomatoes_id'), on='rotten_tomatoes_id',rsuffix = 'imdb_id2')
task3 = task3.dropna(subset= ['audience_average'])


# In[11]:


def avg_classification(number):
    return math.ceil(number)


# In[12]:


task3.audience_average = task3.audience_average.apply(avg_classification)


# In[13]:


score_profit = (task3.groupby('audience_average').agg('sum').made_profit/task3.groupby('audience_average').agg('count').made_profit).fillna(value = 0)


# In[14]:


plt.plot(np.linspace(1, 5, 5, dtype=np.int),score_profit)
plt.xlabel('score')
plt.ylabel('probability of making profit')
plt.title('score vs making profit graph')
plt.savefig('score-profit-graph.png')
plt.show()

