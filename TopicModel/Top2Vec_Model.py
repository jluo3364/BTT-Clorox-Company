from .TopicModel import TopicModel
import numpy as np
import pandas as pd
from top2vec import Top2Vec

from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procedure import *

class Top2Vec_Model(TopicModel):
    def __init__(self, df):
        super().__init__(model_name='top2vec', df=df)

    def train_model(self, subcategories): 
        for subcategory in subcategories:
            self.train_model_subcategory(subcategory) 
        return self.df
    
    def train_model_subcategory(self, subcategory, verbose=True):
        subset_df = self.df[self.df['subcategory'] == subcategory]
        self.df['top2vec_topic_number'] = -1
        self.df['top2vec_topic_words'] = ''
        self.df['top2vec_topic_label'] = ''
        self.df['top2vec_similarity_score'] = -1
        subset_df['top2vec_topic_number'] = -1
        subset_df['top2vec_topic_words'] = ''
        subset_df['top2vec_topic_label'] = ''
        subset_df['top2vec_similarity_score'] = -1
        
        star_rating_dataframes = []
        for i in range(1, 6):
            star_rating_dataframes.append(subset_df[subset_df['star_rating'] == i])

        num_topics_to_generate = get_number_topics_subcategory(subset_df.shape[0])
        desired_num_topics = []
        for i in range(1, 6):
            desired_num_topics.append(calculate_num_topics_star_rating(num_topics_to_generate, i))
        
        # Perform Hierarchical Topic Reduction if necessary
        top2vec_models = []
        topic_reduction = [False]*5
        for i in range(len(star_rating_dataframes)):
            star_rating_dataframe = star_rating_dataframes[i]
            if verbose:
                print(f"Creating Top2Vec model with {desired_num_topics[i]} topics for {star_rating_dataframe.shape[0]} reviews (star rating {i+1}) in subcategory {subcategory}")
            top2vec_model = Top2Vec(documents=list(star_rating_dataframe['review_text']), embedding_model='all-MiniLM-L6-v2', workers=20, ngram_vocab=True)
            # top2vec_model.save("top2vec_saved_models/top2vec_" + str(star_rating_dataframe['star_rating'].iloc[0]) + "_star")
            topics_generated = top2vec_model.get_num_topics()
            if topics_generated > desired_num_topics[i]:
                # do topic reduction 
                topic_reduction[i] = True
                top2vec_model.hierarchical_topic_reduction(desired_num_topics[i])
            top2vec_models.append(top2vec_model)

        # Generate topic labels
        refined_topics = {}
        for i in range(len(top2vec_models)):
            topic_words, word_scores, topic_nums = top2vec_models[i].get_topics(reduced=topic_reduction[i])
            for top_50_topic_words in topic_words:
                top_15_topic_words = top_50_topic_words[:15]
                if refined_topics.get(i+1) is None:
                    refined_topics[i+1] = []
                refined_topics[i+1].append(generate_topic_label(top_15_topic_words, i+1)) # i+1 because ratings are 1-indexed
        
        # Assign topics to each review and add relevant columns to df
        for i in range(len(top2vec_models)):
            print("STAR RATING ", i+1)
            print("COLUMNS", star_rating_dataframes[i].columns)
            topic_nums, topic_score, topic_words, word_scores = top2vec_models[i].get_documents_topics(doc_ids=top2vec_models[i].document_ids, reduced=topic_reduction[i])
            star_rating_dataframes[i]['top2vec_topic_number'] = topic_nums
            star_rating_dataframes[i]['top2vec_topic_label'] = star_rating_dataframes[i]['top2vec_topic_number'].apply(lambda x: refined_topics[i+1][x])
            star_rating_dataframes[i]['top2vec_topic_words'] = star_rating_dataframes[i]['top2vec_topic_number'].apply(lambda x: str(topic_words[x][:15]))
            # star_rating_dataframes[i].reset_index(drop=True, inplace=True)

        # Combine dataframes
        new_df = pd.concat(star_rating_dataframes)
        # replace rows in self.df with the rows in new_df
        self.df.loc[new_df.index] = new_df
        # Calculate similarity scores
        self.calculate_similarity_score(subcategory, True)
        self.calculate_similarity_score(subcategory, False)

        for i in range(len(top2vec_models)):
            self.models[i+1] = top2vec_models[i]
        
        return self.df