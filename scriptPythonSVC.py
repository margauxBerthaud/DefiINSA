#!/usr/bin/env python
# coding: utf-8

# In this notebook we describe the code use to produce the baseline. 

# # Data and Libraries

# In[1]:


import pandas as pd
import unicodedata 
import time
import nltk
import re 
import warnings
from tqdm import tqdm
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

from string import digits
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style("whitegrid")


DATA_PATH = "defi-ia-insa-toulouse"
train_df = pd.read_json(DATA_PATH+"/train.json")
test_df = pd.read_json(DATA_PATH+"/test.json")
train_label = pd.read_csv(DATA_PATH+"/train_label.csv")
label_metier= pd.read_csv(DATA_PATH+"/categories_string.csv")
print("Taille des données train_df : %d lignes" %train_df.shape[0])
print("Taille des données train_label : %d lignes" %train_label.shape[0])
print("Taille des données test : %d lignes" %test_df.shape[0])
print("Taille des labels métiers : %d lignes" %label_metier.shape[0])


# In[2]:


label_metier.rename(columns={"1" : "Category"}, inplace=True)

label_string=pd.merge(train_label,label_metier, on=['Category'])


# On a merge les deux dataframe afin de mieux pouvoir visualiser les métiers en surreprésentation.

# In[3]:


data_count = label_string["0"].value_counts()

fig = go.Figure()
fig.add_trace(go.Bar(x=data_count.index,
                y=data_count.values,
                marker_color='rgb(55, 83, 109)'
                ))

fig.update_layout(
    title='Distribution des description dans les catégories',
    xaxis_tickfont_size=12,
    xaxis_tickangle=70,
    yaxis=dict(
        title='Nombre de descriptions',
        titlefont_size=16,
        tickfont_size=14,
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    barmode='group',
    bargap=0.15, 
    bargroupgap=0.1 
)
fig.show()


# Faut équilibrer classe et genre métier.

# On peut remarquer une grande disparité dans la représentation des catégories.

# In[9]:


desc_metier=pd.merge(label_string,train_df, on=['Id'])


# In[10]:


vocabulary_size = {metier : len(set(" ".join(desc_metier[desc_metier["0"]==metier]["description"].values).split(" "))) for metier in set(desc_metier["0"].values)}

fig = go.Figure()
fig.add_trace(go.Bar(x=data_count.index,
                y=[vocabulary_size[c] for c in data_count.index],
                marker_color='rgb(55, 83, 109)'
                ))

fig.update_layout(
    title='Size of vocabulary per work',
    xaxis_tickfont_size=12,
    xaxis_tickangle=70,
    yaxis=dict(
        title='Size of vocabulary',
        titlefont_size=16,
        tickfont_size=14,
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# # Cleaning

# In[11]:


i = 47
description = train_df.description.values[i]
print("Original Description : " + description)


# In[12]:


digits_list = digits
class CleanText:

    def __init__(self):

        english_stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords = [self.remove_accent(sw) for sw in english_stopwords]

        self.stemmer = nltk.stem.SnowballStemmer('english')

    @staticmethod
    def remove_html_code(txt):
        txt = BeautifulSoup(txt, "html.parser", from_encoding='utf-8').get_text()
        return txt

    @staticmethod
    def convert_text_to_lower_case(txt):
        return txt.lower()

    @staticmethod
    def remove_accent(txt):
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode("utf-8")

    @staticmethod
    def remove_non_letters(txt):
        return re.sub('[^a-z_]', ' ', txt)

    def remove_stopwords(self, txt):
        return [w for w in txt.split() if (w not in self.stopwords)]

    def get_stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]


# In[13]:


c=CleanText()
def apply_all_transformation(description):
    description = c.remove_html_code(description)
    description = c.convert_text_to_lower_case(description)
    description = c.remove_accent(description)
    description = c.remove_non_letters(description)
    tokens = c.remove_stopwords(description)
    tokens = c.get_stem(tokens)
    return tokens


# In[14]:


# %load solution/clean_dataframe_2.py
def clean_df_column(df,column_name, clean_column_name) :
    df[clean_column_name]=[" ".join(apply_all_transformation(x)) for x in tqdm(df[column_name].values)]


# In[15]:


clean_df_column(train_df, "description", "description_cleaned")
train_df[["description", "description_cleaned"]]


# In[16]:


clean_df_column(test_df, "description", "description_cleaned")
test_df[["description", "description_cleaned"]]


# In[ ]:


from wordcloud import WordCloud
all_descr = " ".join(train_df.description.values)
wordcloud_word = WordCloud(background_color="black", collocations=False).generate_from_text(all_descr)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud_word,cmap=plt.cm.Paired)
plt.axis("off")
plt.title("Wordcloud des données brutes")
plt.show()


# In[ ]:


all_descr_clean_stem = " ".join(train_df.description_cleaned.values)
wordcloud_word = WordCloud(background_color="black", collocations=False).generate_from_text(all_descr_clean_stem)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud_word,cmap=plt.cm.Paired)
plt.axis("off")
plt.title("Wordcloud des données 'propres'")
plt.show()


# # Vectorization
# 
# We use TfidfVectorizer to transform words from text to numerical vector data.  
# 
# More vectorize are available on scikit-learn -> https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
# 
# You also may want to have a look at words embedding methods (Word2vec, Glove, etc..)

# **Test de la méthode TfIDF**

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
transformer = TfidfVectorizer().fit(train_df["description_cleaned"].values)
print("NB features: %d" %(len(transformer.vocabulary_)))
X_train = transformer.transform(train_df["description_cleaned"].values)
X_test = transformer.transform(test_df["description_cleaned"].values)
X_train


# # Learning
#  

# In[ ]:


from sklearn.svm import LinearSVC
start = time.time()
Y_train = train_label.Category.values
model = LinearSVC(penalty='l2',loss='hinge', C=1.0)

model.fit(X_train, Y_train)


# # Prediction

# In[31]:


predictions = model.predict(X_test)
predictions


# # File Generation

# In[32]:


test_df["Category"] = predictions
baseline_file = test_df[["Id","Category"]]
baseline_file.to_csv("baseline3.csv", index=False)
