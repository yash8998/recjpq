#!/usr/bin/env python
# coding: utf-8

# In[1]:


from aprec.api.action import Action
from aprec.api.catalog import Catalog
from aprec.datasets.download_file import download_file

BERT4REC_DATASET_URL = "https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/{}.txt"
BERT4REC_DIR = "data/bert4rec"
VALID_DATASETS = {"ml-1m"}

def get_bert4rec_dataset(dataset):
    if dataset not in VALID_DATASETS:
        raise ValueError(f"unknown bert4rec dataset {dataset}")
    
    dataset_filename = download_file(BERT4REC_DATASET_URL.format(dataset), dataset + ".txt", BERT4REC_DIR)
    
    # Get genre and catalog information
    genres_dict, title_dict = get_movielens1m_genres()
    catalog = get_movielens1m_catalog()
    
    actions = []
    prev_user = None
    current_timestamp = 0
    
    with open(dataset_filename) as input:
        for line in input:
            user, item = [str(id) for id in line.strip().split()]
            if user != prev_user:
                current_timestamp = 0
            prev_user = user
            current_timestamp += 1
            genre = genres_dict.get(item, "Unknown") 
            title = title_dict.get(item, "Unknown")# Get genre for the item
            actions.append(Action(user, item, current_timestamp, {"genres":genre,"title":title}))  # Add genre to the action
        return actions
            


MAPPING_URL = "https://raw.githubusercontent.com/asash/ml1m-sas-mapping/main/sas_to_original_items.txt"

def ml1m_mapping_to_original():
    mapping_filename = download_file(MAPPING_URL, "sas_to_original_items.txt", BERT4REC_DIR)
    result = {}
    for line in open(mapping_filename):
        sas_item, original_item = line.strip().split()
        result[sas_item] = original_item
    return result

def get_movielens1m_genres():
    from aprec.datasets.movielens1m import get_genre_title_dict as get_ml1m_genre_dict
    original_genre_dict, original_title_dict = get_ml1m_genre_dict()
    mapping = ml1m_mapping_to_original()
    result_genre = {}
    result_title = {}
    for sas_item, original_item in mapping.items():
        result_genre[sas_item] = original_genre_dict[original_item]
        result_title[sas_item] = original_title_dict[original_item]
    return result_genre,result_title

def get_movielens1m_catalog():
    from aprec.datasets.movielens1m import get_movies_catalog as get_ml1m_catalog
    original_catalog, movie_genres, movie_titles = get_ml1m_catalog()
    mapping = ml1m_mapping_to_original()
    result = Catalog()
    for sas_item, original_item_id in mapping.items():
        item = original_catalog.get_item(original_item_id)
        item.item_id = sas_item
        result.add_item(item)
    return result


# In[2]:


dataset = get_bert4rec_dataset('ml-1m')


# In[3]:


dataset[1]


# In[4]:


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/sentence-t5-base')

def get_embeddings(genres):
        genre_text = ' '.join(genres)
        embeddings = model.encode(genre_text)
        return embeddings


# In[5]:


import numpy as np
item_to_genre = {}
item_ids = []  # Mapping from item ID to genre list

# Populate item_to_genre with the genres for each unique item
for user in dataset:
    #print(user)
    item_ids.append(int(user.item_id))
    genres = user.data['genres']
    item_to_genre[int(user.item_id)] = genres

unique_items = set(item_ids)
print(len(item_to_genre))




# In[ ]:


from sklearn.decomposition import PCA
from tqdm import tqdm
genre_embeddings = np.zeros((3418, 768))

for item_id in tqdm(unique_items, desc="Processing items"):
    #print(int(item_id))
    genres = item_to_genre[item_id]
    genre_emb = get_embeddings(genres)
    
    genre_embeddings[item_id] = genre_emb
    

# Apply PCA to reduce dimensionality of textual embeddings
pca = PCA(n_components=2)
#combined_text_embeddings = np.vstack((genre_embeddings, title_embeddings))
reduced_text_embeddings = pca.fit_transform(genre_embeddings)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], alpha=0.6)
plt.title("PCA of Genre Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('item_embeddings_pca.png')
plt.show()


# In[ ]:




