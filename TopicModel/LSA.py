# inherit TopicModel class and implement LSA model
from TopicModel import TopicModel
import time
import sys
import os

# Add the directory containing procedure.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procedure import *
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class LSA(TopicModel):
        
    def __init__(self, df):
        super().__init__(model_name='lsa', df=df)
    
    '''
    Creates the LSA model on the given subcategories.
    Returns dataframe with added columns: topic_number, top_15_words, topic_label, similarity_score
    '''
    def train_model(self, subcategories, verbose=0):
        for subcategory in subcategories:
            self.train_model_subcategory(subcategory, verbose=verbose)
        return self.df
    

    '''
    Creates the LSA model components
    '''
    def create_components(self, n_topics):
        vectorizer = TfidfVectorizer(stop_words='english', 
                                 use_idf=True, 
                                 ngram_range=(1, 2),
                                 smooth_idf=True)
        svd_model = TruncatedSVD(n_components=n_topics,        
                                algorithm='randomized',
                                n_iter=20)
        return vectorizer, svd_model
    
    
    '''
    Creates the LSA model on one subcategory.
    Returns dataframe with added columns: topic_number, top_15_words, topic_label, similarity_score
    '''
    def train_model_subcategory(self, subcategory, verbose=0, calc_similarity=True):
        start_overall = time.time()
        subset_df = self.df[self.df['subcategory'] == subcategory]
        rating_reviews = subset_df.groupby('star_rating').apply(lambda x: x['review_text'].tolist()).to_dict()
        rating_indices = subset_df.groupby('star_rating').apply(lambda x: x.index).to_dict()
        total_topics = get_number_topics_subcategory(len(subset_df))

        # create the model for each star rating
        for rating, reviews in rating_reviews.items():
            rating_num_topics = calculate_num_topics_star_rating(total_topics, rating)
            if verbose:
                print(f"Creating LSA model for {rating} star rating with {len(reviews)} reviews, {rating_num_topics} topics")
            start = time.time()
            vectorizer, svd_model = self.create_components(rating_num_topics)

            X = vectorizer.fit_transform(reviews)
            topic_matrix = svd_model.fit_transform(X)

            terms = vectorizer.get_feature_names_out()
            topic_vectors = svd_model.components_

            topic_words_labels =  {}  # {topic_number: [top_words, label]}
            for i in range(rating_num_topics):
                top_word_indices = np.argsort(topic_vectors[i])[-15:]  # get the indices of top 15 words
                topic_words = [terms[idx] for idx in top_word_indices]
                topic_label = generate_topic_label(topic_words, rating)
                if verbose == 2:
                    print(f"Topic {i}: {topic_label}\n\t{topic_words}")
                topic_words_labels[i] = [topic_words, topic_label]
            end = time.time()
            if verbose:
                print(f"Finished creating LSA model for {rating} star rating in {end-start:.2f} seconds")
                print()
                print('-'*50)
            self.models[rating] = (vectorizer, svd_model, topic_matrix)
            topic_for_reviews = topic_matrix.argmax(axis=1)
            model = self.model_name
            cur_reviews = rating_indices[rating]
            self.df.loc[cur_reviews, f'{model}_topic_number'] = topic_for_reviews
            self.df.loc[cur_reviews, f'{model}_top_15_words'] = [', '.join(topic_words_labels[i][0]) for i in topic_for_reviews]
            self.df.loc[cur_reviews, f'{model}_topic_label'] = [topic_words_labels[i][1] for i in topic_for_reviews]
       
        if verbose:
            print(f"Finished creating LSA models for {subcategory} in {time.time()-start_overall:.2f} seconds")
            print('-'*200)

        if calc_similarity:
            if verbose:
                print(f"Calculating similarity scores for {subcategory}")
            self.calculate_similarity_score(subcategory)
        
        return self.df
    