import pandas as pd
import sys
import os

# Add the directory containing procedure.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procedure import similarity_scores

class TopicModel():

    def __init__(self, model_name, df):
        self.model_name = model_name
        self.models = {} # star rating: model obj
        self.df = df # dataframe with initial review data

    '''
    Trains the topic model on the given subcategories.
    Returns dataframe with added columns: topic_number, top_15_words, topic_label, similarity_score
    '''
    def train_model(self, subcategories): 
        # chunk by star rating and subcategories
        # train their model
        # add “{model_name}_topic_number”
        # add “{model_name}_top_15_words” 
        # generate the topic phrases through LLM
        # add  “{model_name}_topic_label” 
        # call similarity score function (import from procedure)
        # add “{model_name}_similarity_score”
        raise NotImplementedError
    
    '''
    Calculates the similarity score between reviews and topics
    '''
    def calculate_similarity_score(self, subcategory):
        reviews = self.df[self.df['subcategory'] == subcategory]['review_text'].tolist()
        topic_labels = self.df[self.df['subcategory'] == subcategory][f'{self.model_name}_topic_label'].tolist()
        self.df.loc[self.df['subcategory'] == subcategory, f'{self.model_name}_similarity_score'] = similarity_scores(reviews, topic_labels)
        return self.df

    '''
    Returns a list of dataframes, one for each star rating
    '''
    def get_star_rating_dataframes(self):
        star_rating_dataframes = []
        for i in range(1, 6):
            star_rating_dataframes.append(self.df[self.df['star_rating'] == i])
        return star_rating_dataframes

    '''
    Returns the list of all five models
    '''
    def get_models(self):
        return self.models
    
    '''
    Returns a dataframe with the topic phrase, star rating, count, and average similarity score
    for topics in the given subcategory
    '''
    def get_topic_information(self, subcategory):
        model = self.model_name
        subset_df = self.df[self.df['subcategory'] == subcategory]
        topic_similarity_df = subset_df.groupby(['star_rating', f'{model}_topic_label'])[f'{model}_similarity_score'].mean().reset_index()
        topic_similarity_df['count'] = subset_df.groupby(['star_rating', f'{model}_topic_label'])[f'{model}_similarity_score'].count().values
        return topic_similarity_df
    
