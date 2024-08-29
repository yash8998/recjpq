# Overview

This project explores the enhancement of the RecJPQ model by integrating content embeddings to improve recommendation accuracy in sequential recommendation tasks. By leveraging both collaborative filtering and content-based filtering, the hybrid system aims to overcome the limitations of traditional recommendation models, such as the cold start problem and data sparsity.

# Features
Hybrid Embeddings: Combines collaborative and content embeddings to create a more comprehensive item representation.
Efficient Embedding Compression: Utilizes the RecJPQ model, which adapts Joint Product Quantization (JPQ) for sequential recommendations, reducing the computational burden.
Sequential Recommendation: Supports advanced sequential recommendation models like SASRec and BERT4Rec, enhanced with compressed hybrid embeddings.
Scalability: Designed to handle large-scale item catalogs by compressing high-dimensional embeddings into low-dimensional representations.

# Datasets
The project primarily uses the MovieLens 1M dataset, which includes:

1 million ratings from 6,000 users on 4,000 movies.
User demographic information and movie metadata (e.g., titles, genres).
Additional genre data sourced from The Movie Database (TMDB) to enrich content embeddings.

# Instruction

The code is based on the aprec framework by Petrov & Macdonald, (2023) . Please clone this code and follow the original instructions https://github.com/asash/bert4rec_repro to setup the environment. 

The code for the RecJPQ versions of the model described in the paper is located in the folder recommenders/sequential/models/recjpq. 
Configuration files can be found in evaluation/configs/jpq

Please follow the instructions https://github.com/asash/bert4rec_repro to run the experiments. 

