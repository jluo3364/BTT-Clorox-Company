# inherit TopicModel class and implement LSA model
try:
    # When executed as part of a package (e.g., via main.py)
    from .TopicModel import TopicModel
except ImportError:
    # When executed directly or in a Jupyter notebook
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

class BERTopic_kmeans(TopicModel):
    def __init__(self, df):
        super().__init__('bertopic_kmeans', df)

    def train_model(self, subcategories, verbose=0, top_n_words=15):
        for subcategory in subcategories:
            self.train_model_subcategory(subcategory, verbose=verbose, top_n_words=top_n_words)
        return self.df

    def create_topic_model(self, text, num_topics, top_n_words=15):
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
            top_n_words=top_n_words,                           # Number of top words per topic
            embedding_model=embedding_model,          # Step 1 - Extract embeddings
            umap_model=pca_model,                    # Step 2 - Reduce dimensionality
            hdbscan_model=cluster_model,              # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
            representation_model=representation_model # Step 6 - (Optional) Fine-tune topic representations
        )

        # Fit BERTopic model
        topic_model.fit_transform(text)
        return topic_model
    
    '''
    Creates the bertopic model on one subcategory.
    Returns dataframe with added columns: topic_number, topic_words, topic_label, similarity_score
    '''
    def train_model_subcategory(self, subcategory, verbose=0, calc_similarity=True, top_n_words=15):
        model_capitalized = "BERTopic kmeans" 
        print(f"\nCreating {model_capitalized} models for {subcategory}")
        start_overall = time.time()
        subset_df = self.df[self.df['subcategory'] == subcategory]
        rating_reviews = subset_df.groupby('star_rating').apply(lambda x: x['review_text'].tolist()).to_dict()
        rating_indices = subset_df.groupby('star_rating').apply(lambda x: x.index).to_dict()
        total_topics = get_number_topics_subcategory(len(subset_df))
        model_name = self.model_name
        # df_topics = pd.DataFrame(columns=['star_rating', f'{model_name}_topic_number', f'{model_name}_topic_words'])  # dataframe to store topic info

        # create the model for each star rating
        for rating, reviews in rating_reviews.items():
            rating_num_topics = calculate_num_topics_star_rating(total_topics, rating)
            if verbose:
                print(f"Creating {model_capitalized} model for {rating} star rating with {len(reviews)} reviews, {rating_num_topics} topics")
            start = time.time()
            model = self.create_topic_model(reviews, rating_num_topics, top_n_words=top_n_words)
            
            output = model.get_topic_info()  # topic info for current rating's model
            topic_words_labels =  {}  # {topic_number: [top_words, label]}
            for _, row in output.iterrows():
                topic_words = row['Representation']
                topic_label = generate_topic_label(topic_words, rating)
                topic_label = self.clean_topic_label(topic_label)
                topic_words_labels[row['Topic']] = [topic_words, topic_label]
                if verbose == 2:
                    print(f"Topic {row['Topic']}: {topic_label}\n\t{topic_words}")
            
            if verbose:
                print(f"Finished creating {model_capitalized} model for {rating} in {time.time() - start:.2f} seconds")
                print('\n' +'-'*50)
            #     new_row = pd.DataFrame({
            #         'star_rating': [rating],
            #         f'{model_name}_topic_number': [row['Topic']],
            #         f'{model_name}_topic_words': [row['Representation']]
            #     })
            #     df_topics = pd.concat([df_topics, new_row], ignore_index=True)  # add topics for current rating to df_topics
           
           # get df of current reviews with topic labels
            cur_reviews_indices = rating_indices[rating]
            labeled_reviews = model.get_document_info(reviews)
            labeled_reviews.loc[:, 'Representation'] = labeled_reviews['Representation'].map(lambda x: ', '.join(x))
            self.df.loc[cur_reviews_indices, f'{model_name}_topic_number'] = [i for i in labeled_reviews['Topic']]
            self.df.loc[cur_reviews_indices, f'{model_name}_topic_words'] = [words for words in labeled_reviews['Representation']]
            self.df.loc[cur_reviews_indices, f'{model_name}_topic_label'] = [topic_words_labels[i][1] for i in labeled_reviews['Topic']]
            if self.models.get(subcategory) is None:
                self.models[subcategory] = {}
            if self.models.get(subcategory) is None:
                self.models[subcategory] = {}
            self.models[subcategory][rating] = model  # add model to models dictionary
        
        if verbose:
            print(f"Finished creating {model_capitalized} models for {subcategory} in {time.time()-start_overall:.2f} seconds")
            print('-'*200)

        if calc_similarity:
            if verbose:
                print(f"Calculating similarity scores for {subcategory}")
            self.calculate_similarity_score(subcategory)
        return self.df

    