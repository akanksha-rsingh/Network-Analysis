#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx
from operator import itemgetter
import matplotlib.pyplot
import pandas as pd


# In[2]:


# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('./amazon-books.csv', index_col=0)


# In[3]:


# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('./amazon-books.csv', index_col=0)

# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=networkx.read_weighted_edgelist(fhr)
fhr.close()


# In[4]:


# Now let's assume a person is considering buying the following book;
# what else can we recommend to them based on copurchase behavior 
# we've seen from other users?
print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'


# In[5]:


# Let's first get some metadata associated with this book
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks.loc[purchasedAsin,'Title'])
print ("SalesRank = ", amazonBooks.loc[purchasedAsin,'SalesRank'])
print ("TotalReviews = ", amazonBooks.loc[purchasedAsin,'TotalReviews'])
print ("AvgRating = ", amazonBooks.loc[purchasedAsin,'AvgRating'])
print ("DegreeCentrality = ", amazonBooks.loc[purchasedAsin,'DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks.loc[purchasedAsin,'ClusteringCoeff'])


# In[6]:


# Now let's look at the ego network associated with purchasedAsin in the
# copurchaseGraph - which is esentially comprised of all the books 
# that have been copurchased with this book in the past
# (1) YOUR CODE HERE: 
#     Get the depth-1 ego network of purchasedAsin from copurchaseGraph,
#     and assign the resulting graph to purchasedAsinEgoGraph.
purchasedAsinEgoGraph = networkx.ego_graph(copurchaseGraph, purchasedAsin,radius=1)


# In[7]:


# Next, recall that the edge weights in the copurchaseGraph is a measure of
# the similarity between the books connected by the edge. So we can use the 
# island method to only retain those books that are highly simialr to the 
# purchasedAsin
# (2) YOUR CODE HERE: 
#     Use the island method on purchasedAsinEgoGraph to only retain edges with 
#     threshold >= 0.5, and assign resulting graph to purchasedAsinEgoTrimGraph
threshold = 0.5
purchasedAsinEgoTrimGraph = networkx.Graph()
for f, t, e in purchasedAsinEgoGraph.edges(data=True):
    if e['weight'] >= threshold:
        purchasedAsinEgoTrimGraph.add_edge(f,t,weight=e['weight'])


# In[8]:


# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) YOUR CODE HERE: 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = []
for f, t, e in purchasedAsinEgoTrimGraph.edges(data=True):
    if f == purchasedAsin:
        purchasedAsinNeighbors.append(t)
print(purchasedAsinNeighbors)


# In[9]:


threshold = 0.5
purchasedAsinEgoTrimGraph = networkx.Graph()
for f,t,e in purchasedAsinEgoGraph.edges(data=True):
    if e['weight'] >= threshold:
        purchasedAsinEgoTrimGraph.add_edge(f,t, weight=e['weight'])


# In[10]:


# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) YOUR CODE HERE: 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = [i for i in purchasedAsinEgoTrimGraph.neighbors(purchasedAsin)]
print(purchasedAsinNeighbors)


# In[11]:


# Next, let's pick the Top Five book recommendations from among the 
# purchasedAsinNeighbors based on one or more of the following data of the 
# neighboring nodes: SalesRank, AvgRating, TotalReviews, DegreeCentrality, 
# and ClusteringCoeff
# (4) YOUR CODE HERE: 
#     Note that, given an asin, you can get at the metadata associated with  
#     it using amazonBooks (similar to lines 29-36 above).
#     Now, come up with a composite measure to make Top Five book 
#     recommendations based on one or more of the following metrics associated 
#     with nodes in purchasedAsinNeighbors: SalesRank, AvgRating, 
#     TotalReviews, DegreeCentrality, and ClusteringCoeff. Feel free to compute
#     and include other measures if you like.
#     YOU MUST come up with a composite measure.
#     DO NOT simply make recommendations based on sorting!!!
#     Also, remember to transform the data appropriately using 
#     sklearn preprocessing so the composite measure isn't overwhelmed 
#     by measures which are on a higher scale.


# In[12]:


df = pd.DataFrame()
for i in range(len(purchasedAsinNeighbors)):
    df1 = amazonBooks.loc[[purchasedAsinNeighbors[i]],['Title','SalesRank','AvgRating','TotalReviews','DegreeCentrality','ClusteringCoeff']]
    df = pd.concat((df,df1),axis=0)
df


# In[13]:


from sklearn.preprocessing import MinMaxScaler

numeric_vars = ['SalesRank', 'AvgRating', 'TotalReviews', 'DegreeCentrality', 'ClusteringCoeff']

mms = MinMaxScaler()
dfnumss = pd.DataFrame(mms.fit_transform(df[numeric_vars]), columns=['mss_'+x for x in numeric_vars], index = df.index)
dfnumss = pd.concat([df, dfnumss], axis=1)
dfnumss = dfnumss.drop(numeric_vars, axis=1)
dfnumss.head()


# In[14]:


df = dfnumss.copy()


# In[15]:


df.head()


# In[17]:


df.describe()


# In[43]:


#CompositeScore = SalesRank less than mean,Avg rating greater than mean, Total reviews greater than mean, Clustercoeff greater than 25 percentile
CompositeScoreDF = df[(df['mss_SalesRank']<=0.202) & (df['mss_AvgRating']>=0.9)& (df['mss_TotalReviews']>=0.24)& (df['mss_ClusteringCoeff']>=0.11)]
CompositeScoreDF


# In[44]:


amazonbooks = amazonBooks[['SalesRank','TotalReviews','AvgRating','DegreeCentrality','ClusteringCoeff']]
amazonbooks.head()


# In[45]:


MyRecommendations = pd.concat([CompositeScoreDF,amazonbooks],axis=1,join='inner')
MyRecommendations = MyRecommendations.drop(['mss_SalesRank','mss_AvgRating','mss_TotalReviews','mss_DegreeCentrality','mss_ClusteringCoeff'],axis = 1)
MyRecommendations


# In[ ]:




