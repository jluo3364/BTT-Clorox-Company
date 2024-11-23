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
    Returns dataframe with added columns: topic_number, topic_words, topic_label, similarity_score
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
    def calculate_similarity_score(self, subcategory, topic_words=False):
        if topic_words:
            reviews = self.df[self.df['subcategory'] == subcategory]['review_text'].tolist()
            topic_words = self.df[self.df['subcategory'] == subcategory][f'{self.model_name}_topic_words'].tolist()
            self.df.loc[self.df['subcategory'] == subcategory, f'{self.model_name}_words_similarity_score'] = similarity_scores(reviews, topic_words)
        else:
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
    
    def clean_topic_label(self, topic_label):
        # if generated topic label starts with 'Here is a concise and coherent phrase', usually real topic label is in double quotes
        try:
            if topic_label.startswith('Here is a'):
                topic_label = topic_label.split(':')[1]
        except:
            pass
        # remove double quotes
        topic_label = topic_label.replace('"', '')
        return topic_label
    
    @staticmethod
    def subcategories_of_size(df, target_sizes, num_subcategories, allowance=1000):
        """
        df: dataframe, the data to find subcategories from.
        target_sizes = list of integers, sizes to find subcategories for.
        num_subcategories: list of integers, parallel to target_sizes and indicates number of subcategories to return for each target_size.
        allowance: integer, how much to allow the size to vary from the target_size.
        Returns a list of subcategories that have sizes that are target_sizes +/- allowance.
        """
        subcat_sizes = df['subcategory'].value_counts().to_dict()
        subcategories = []
        for target_size in target_sizes:
            valid_subcats = [name for name, size in subcat_sizes.items() if abs(size - target_size) < allowance]
            subcategories.extend(valid_subcats[:num_subcategories[target_sizes.index(target_size)]])
        return subcategories