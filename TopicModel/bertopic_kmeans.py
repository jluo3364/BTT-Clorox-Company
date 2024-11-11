from TopicModel import TopicModel
import time
import sys
import os

# Add the directory containing procedure.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procedure import *
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.decomposition import PCA

class BERTopic(TopicModel):
    def __init__(self, df):
        super().__init__('bertopic_kmeans', df)


    '''
    Creates the bertopic model on one subcategory.
    Returns dataframe with added columns: topic_number, top_15_words, topic_label, similarity_score
    '''
    def train_model_subcategory(self, subcategory, verbose=0, calc_similarity=True):
        print(f"\nCreating BERTopic models for {subcategory}")
        start_overall = time.time()
        subset_df = self.df[self.df['subcategory'] == subcategory]
        rating_reviews = subset_df.groupby('star_rating').apply(lambda x: x['review_text'].tolist()).to_dict()
        rating_indices = subset_df.groupby('star_rating').apply(lambda x: x.index).to_dict()
        total_topics = get_number_topics_subcategory(len(subset_df))

        # create the model for each star rating
        for rating, reviews in rating_reviews.items():
            rating_num_topics = calculate_num_topics_star_rating(total_topics, rating)
            if verbose:
                print(f"Creating BERTopic model for {rating} star rating with {len(reviews)} reviews, {rating_num_topics} topics")
            start = time.time()
            
    def create_topic_model(text, num_topics):
        # Step 1 - Extract embeddings
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Step 2 - Reduce dimensionality
        pca_model = PCA(n_components=10)

        # Step 3 - Cluster reduced embeddings
        cluster_model = KMeans(n_clusters=num_topics)

        # Step 4 - Tokenize topics
        vectorizer_model = CountVectorizer(stop_words="english")

        # Step 5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()

        # Step 6 - (Optional) Fine-tune topic representations with 
        # a `bertopic.representation` model
        representation_model = KeyBERTInspired()

        topic_model = BERTopic(
            embedding_model=embedding_model,          # Step 1 - Extract embeddings
            umap_model=pca_model,                    # Step 2 - Reduce dimensionality
            hdbscan_model=cluster_model,              # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model                # Step 5 - Extract topic words
        # representation_model=representation_model # Step 6 - (Optional) Fine-tune topic representations
        )

        # Fit BERTopic model
        topic_model.fit_transform(text)
        return topic_model