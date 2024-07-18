from lightfm import LightFM
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np


class BPRWithContextAssignmentStrategy(CentroidAssignmentStragety):

    def get_embeddings(self, genres):
        genre_text = ' '.join(genres)
        inputs = self.tokenizer(genre_text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    def assign(self, train_users):
        rows = []
        cols = []
        vals = []

        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j][1])
                vals.append(1)

        matr = csr_matrix((vals, (rows, cols)), shape=(len(train_users), self.num_items+2))

        unique_items = set(cols)  # Get unique item IDs
        item_to_genre = {}  # Mapping from item ID to genre list
        movie_titles = {}

        # Populate item_to_genre with the genres for each unique item
        for user in train_users:
            for interaction in user:
                print(interaction)
                item_id = interaction[1]
                genres = interaction[2]
                titles = interaction[3]
                item_to_genre[item_id] = genres
                movie_titles[item_id] = titles

        genre_embeddings = np.zeros((self.num_items+2, 768))
        title_embeddings = np.zeros((self.num_items + 2, 768))

        for item_id in unique_items:
            genres = item_to_genre[item_id]
            genre_emb = self.get_embeddings(genres)
            # Store embedding at index corresponding to item_id
            genre_embeddings[item_id] = genre_emb
            title_embeddings[item_id] = self.get_embeddings(movie_titles[item_id])

        print("Fitting MF-BPR for initial centroids assignments")
        model = LightFM(no_components=self.item_code_bytes, loss='bpr')
        model.fit(matr, epochs=20, verbose=True, num_threads=20)
        item_embeddings = model.get_item_representations()[1].T
        genre_embeddings = genre_embeddings.T
        title_embeddings = title_embeddings.T
        print("Item embeddings shape:", item_embeddings.shape)
        print("Genre embeddings matrix shape:", genre_embeddings.shape)
        print(title_embeddings)
        combined_embeddings = np.vstack((item_embeddings, genre_embeddings,title_embeddings))

        assignments = []
        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
            ith_component = combined_embeddings[i:i + 1][0]
            ith_component = (ith_component - np.min(ith_component)) / (
                        np.max(ith_component) - np.min(ith_component) + 1e-10)
            noise = np.random.normal(0, 1e-5, self.num_items + 2)
            ith_component += noise  # make sure that every item has unique value
            ith_component = np.expand_dims(ith_component, 1)
            component_assignments = discretizer.fit_transform(ith_component).astype('uint8')[:, 0]
            assignments.append(component_assignments)
            print(np.transpose(np.array(assignments)).shape)
        return np.transpose(np.array(assignments))