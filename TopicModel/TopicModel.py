import pandas as pd

class TopicModel():

    def __init__(self, model_name, df):
        self.model_name = model_name
        self.models = [] # List of model objects, one per star rating
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
    '''
    def get_topic_information(self):
        topic_similarity_df = self.df.groupby([f'{self.model_name}_topic', 'star_rating'])['similarity_score'].mean().reset_index()
        topic_similarity_df['count'] = self.df.groupby([f'{self.model_name}_topic', 'star_rating'])['similarity_score'].count().values
        return topic_similarity_df
    
