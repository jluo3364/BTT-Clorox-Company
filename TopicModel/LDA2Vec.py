from .TopicModel import TopicModel
import pandas as pd
import numpy as np

import spacy

from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans
from collections import Counter

from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

import time

# import matplotlib.pyplot as plt
# import seaborn as sns

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procedure import *

# Load spacy model
nlp = spacy.load('en_core_web_sm')

class LDA2Vec(TopicModel):

    def __init__(self, df):
        super().__init__(model_name='lda2vec', df=df)
    
    def train_model(self, subcategories, verbose=True):
        for subcategory in subcategories:
            self.train_model_subcategory(subcategory, verbose) 
        return self.df
    
    def train_model_subcategory(self, subcategory, verbose=True, calc_similarity=True):

        start_time_overall = time.time()
        subset_df = self.df[self.df['subcategory'] == subcategory]
        subset_df['lda2vec_topic_id'] = -1
        subset_df['lda2vec_topic_label'] = ''
        subset_df['lda2vec_topic_words'] = ''

        grouped = subset_df.groupby('star_rating')
        # subcategory size 
        subcategory_size = subset_df[subset_df['subcategory']==subcategory].shape[0]

        # get total number of topics for the entire subcategory
        total_topics = get_number_topics_subcategory(subcategory_size)

        # Loop through each (subcategory, star_rating) group
        for (star_rating), group in grouped:
            # STEP 1: Train LDA model to get topic distributions for each document
            # Preprocess the documents

            start_time = time.time()
            def create_custom_stopwords(df):
                unique_words = set()
                df['product_title'].drop_duplicates().str.split().apply(unique_words.update)
                df['brand'].drop_duplicates().dropna().str.split().apply(unique_words.update)
                unique_words = list(unique_words)

                custom_stopwords = unique_words + ['nt'] + [word.lower() for word in unique_words] + [
                    word.upper() for word in unique_words] + [word.capitalize() for word in unique_words]
                custom_stopwords = list(set(custom_stopwords))

                return custom_stopwords
            
            custom_stopwords = create_custom_stopwords(group)
            texts = [[word.text for word in nlp(doc.lower()) if not word.is_stop and word.text not in custom_stopwords] for doc in group['review_text']]

            # Create a dictionary and corpus for LDA
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

        
            # Calculate the number of topics based on the size of the group
             # size of current subcategory and star rating
            group_size = group.shape[0]
            num_topics = calculate_num_topics_star_rating(total_topics, star_rating)
            if verbose:
                print(f"Creating LDA2Vec Model with {num_topics} topics for {group_size} reviews (star rating {star_rating}) in subcategory {subcategory}")

            # Train the LDA model
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)

            # Get the topic distribution for each document
            doc_topic_distributions = [lda_model.get_document_topics(bow) for bow in corpus]
            
            ## STEP 2: train word2vec model
            word2vec_model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, sg=1, epochs=50)
            word_vectors = word2vec_model.wv

            ## STEP 3: Convert topic distributions to vectors
            # Create a matrix to hold topic embeddings (you can initialize randomly or train separately)
            topic_embeddings = np.random.rand(lda_model.num_topics, word2vec_model.vector_size)

            # Function to convert a topic distribution to a vector
            def get_topic_vector(topic_distribution):
                vec = np.zeros(word2vec_model.vector_size)
                for topic_id, weight in topic_distribution:
                    vec += weight * topic_embeddings[topic_id]
                return vec

            # Get topic vectors for each document
            doc_topic_vectors = [get_topic_vector(dist) for dist in doc_topic_distributions]

            ## STEP 4: Combine topic vectors and word vectors
            # Function to get average word vector for a document
            def get_word_vector(text):
                words = [word for word in text if word in word_vectors]
                if words:
                    return np.mean([word_vectors[word] for word in words], axis=0)
                else:
                    return np.zeros(word2vec_model.vector_size)

            # Combine topic vector and word vector to get a document vector
            doc_vectors = []
            for text, topic_vec in zip(texts, doc_topic_vectors):
                word_vec = get_word_vector(text)
                doc_vec = word_vec + topic_vec  # Combining topic and word vectors
                doc_vectors.append(doc_vec)
                
            # STEP 5: Cluster the Document Vectors
            kmeans = KMeans(n_clusters=num_topics, random_state=42)
            cluster_labels = kmeans.fit_predict(doc_vectors)
            
            # STEP 6: Analyze Topics in Each Cluster
            
            lda2vec_topic_labels = []
            lda2vec_topic_words = []
            for cluster_id in range(num_topics):

                # Filter documents belonging to this cluster
                cluster_docs = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]

                # Get top 15 most frequent words in this cluster for interpretation
                all_words = [word for doc in cluster_docs for word in doc]
                most_common_words = Counter(all_words).most_common(15)
                common_words = ", ".join([word for word, freq in most_common_words])
                lda2vec_topic_words.append(common_words)
                    
                # Refine the topic label using Groq API
                lda2vec_topic_label = generate_topic_label(common_words, star_rating)
                lda2vec_topic_labels.append(lda2vec_topic_label)
            # print(lda2vec_topic_labels)


            # STEP 7: Map topic words and labels back to DataFrame
            for idx, doc_id in enumerate(group.index):
                cluster_id = cluster_labels[idx]
                lda2vec_topic_label = lda2vec_topic_labels[cluster_id]
                lda2vec_topic_words_text = lda2vec_topic_words[cluster_id]

                # Calculate similarity scores between review and topic words using similarity_scores function
                # lda2vec_words_similarity_score = similarity_scores('all-MiniLM-L6-v2', [group.loc[doc_id, 'review_text']], [lda2vec_topic_words_text])[0]
                # # print(lda2vec_words_similarity_score)

                # # Calculate similarity scores between review and topic label using similarity_scores function
                # lda2vec_label_similarity_score = similarity_scores('all-MiniLM-L6-v2', [group.loc[doc_id, 'review_text']], [lda2vec_topic_label])[0]

                # Assign values to DataFrame 
                self.df.at[doc_id, 'lda2vec_topic_id'] = cluster_id
                self.df.at[doc_id, 'lda2vec_topic_label'] = lda2vec_topic_label
                self.df.at[doc_id, 'lda2vec_topic_words'] = lda2vec_topic_words_text
                # df_sample.at[doc_id, 'lda2vec_words_similarity_score'] = lda2vec_words_similarity_score
                # df_sample.at[doc_id, 'lda2vec_label_similarity_score'] = lda2vec_label_similarity_score
            end_time = time.time()

            if verbose:
                print(f"Finished creating LDA2Vec model for {star_rating} star rating in {end_time-start_time:.2f} seconds")
                print()
                print('-'*50)
        
        end_time_overall = time.time()
        if verbose:
            print(f"Finished LDA2Vec topic modeling for subcategory {subcategory} in {end_time_overall-start_time_overall:.2f} seconds")
            print()
            print('-'*200)
        
        # calculate similarity score
        if calc_similarity:
            self.calculate_similarity_score(subcategory, True)
            self.calculate_similarity_score(subcategory, False)
            if verbose:
                print(f"Calculating similarity scores for {subcategory}")
        
        return self.df
