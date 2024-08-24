from lightfm import LightFM
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np
import torch
import time

class ContentEmbeddingsStrategyPCA(CentroidAssignmentStragety):

    def get_embeddings(self, combined_text):
        embeddings = self.model.encode(combined_text)
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

        matr = csr_matrix((vals, (rows, cols)), shape=(len(train_users), self.num_items + 2))
        print(self.num_items + 2)
        unique_items = set(cols)  # Get unique item IDs
        item_to_genre_title = {}  # Mapping from item ID to combined genre and title text

        # Populate item_to_genre_title with the combined genre and title for each unique item
        for user in train_users:
            for interaction in user:
                item_id = interaction[1]
                genres = interaction[2]
                title = interaction[3]  # Assuming the fourth element is the title
                # combined_text = f"{' '.join(genres)} [SEP] {title}"  # Combine genre and title with a separator
                combined_text = f"{' '.join(genres)}"
                item_to_genre_title[item_id] = combined_text

        # Initialize combined text embeddings
        combined_text_embeddings = np.zeros((self.num_items + 2, 768))

        for item_id in unique_items:
            combined_text = item_to_genre_title[item_id]
            combined_emb = self.get_embeddings(combined_text)
            combined_text_embeddings[item_id] = combined_emb

        print("Combined embeddings matrix shape:", combined_text_embeddings.shape)
        print(self.item_code_bytes)

        print("Fitting MF-BPR for initial centroids assignments")
        model = LightFM(no_components=self.item_code_bytes *3 // 4, loss='bpr')
        model.fit(matr, epochs=20, verbose=True, num_threads=20)
        item_embeddings = model.get_item_representations()[1].T
        combined_text_embeddings = combined_text_embeddings.T

        print("Item embeddings shape:", item_embeddings.shape)
        
        start_time = time.time()
        # Dimensionality Reduction with TruncatedSVD
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        combined_text_embeddings_torch = torch.tensor(combined_text_embeddings, device=device)
        U, S, Vt = torch.linalg.svd(combined_text_embeddings_torch, full_matrices=False)

        print("SVD Done")

        k = self.item_code_bytes - int(self.item_code_bytes *3// 4)
        reduced_content_emb = Vt[:k, :]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.4f} seconds")

        
        combined_embeddings = np.vstack((item_embeddings, reduced_content_emb.cpu().numpy()))

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

        return np.transpose(np.array(assignments))
