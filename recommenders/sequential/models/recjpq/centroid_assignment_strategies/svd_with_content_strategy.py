from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

class SVDWithContentAssignmentStrategy(CentroidAssignmentStragety):
    def get_embeddings(self, genres):
        genre_text = ' '.join(genres)
        embeddings = self.model.encode(genre_text)
        return embeddings
        
    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j][1])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.num_items+2)) # +2 for padding and mask
        print("fitting svd for initial centroids assignments")
        svd = TruncatedSVD(n_components=self.item_code_bytes-2)
        svd.fit(matr)
        item_embeddings = svd.components_

        item_to_genre = {}  # Mapping from item ID to genre list
        movie_titles = {}

        # Populate item_to_genre with the genres for each unique item
        for user in train_users:
            for interaction in user:
                print(interaction)
                item_id = interaction[1]
                genres = interaction[2]
                item_to_genre[item_id] = genres

        genre_embeddings = np.zeros((self.num_items+2, 768))

        for item_id in unique_items:
            genres = item_to_genre[item_id]
            genre_emb = self.get_embeddings(genres)
            # Store embedding at index corresponding to item_id
            genre_embeddings[item_id] = genre_emb

        # Apply PCA to reduce dimensionality of textual embeddings
        pca = PCA(n_components=2)
        reduced_text_embeddings = pca.fit_transform(genre_embeddings)
        genre_embeddings = reduced_text_embeddings[:self.num_items+2]

        combined_embeddings = np.vstack((item_embeddings, genre_embeddings))
        assignments = []
        print("done")
        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
            ith_component = combined_embeddings[i:i+1][0]
            ith_component = (ith_component - np.min(ith_component))/(np.max(ith_component) - np.min(ith_component) + 1e-10)
            noise = np.random.normal(0, 1e-5, self.num_items + 2)
            ith_component += noise # make sure that every item has unique value
            ith_component = np.expand_dims(ith_component, 1)
            component_assignments = discretizer.fit_transform(ith_component).astype('uint8')[:,0]
            assignments.append(component_assignments)
        return np.transpose(np.array(assignments))
