import math
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import os
from groq import Groq 
import time
import string
import nltk
from nltk.stem import WordNetLemmatizer



def lemmatize_text(text):
     # Download required NLTK data
     nltk.download('wordnet')
     nltk.download('punkt_tab')
     nltk.download('averaged_perceptron_tagger')
     nltk.download('averaged_perceptron_tagger_eng')
     lemmatizer = WordNetLemmatizer()
     # Tokenize the text
     words = nltk.word_tokenize(text)

     # Perform part-of-speech tagging
     pos_tags = nltk.pos_tag(words)

     # Lemmatize words based on their part of speech
     lemmatized_words = []
     for word, pos in pos_tags:
        if pos.startswith('N'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='n'))
        elif pos.startswith('V'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))
        elif pos.startswith('R'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='r'))
        elif pos.startswith('J'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='a'))
        else:
            lemmatized_words.append(word)
    
     return ' '.join(lemmatized_words)

def preprocess_review_text(df, review_col='review_text', drop_duplicates=True):
     """
     df: a pandas DataFrame containing a column 'review_text' with the raw review text
     review_col: the name of the column containing the review text
     drop_duplicates: a boolean indicating whether to drop rows that have duplicate review_text
     """

     df.loc[:, 'og_'+review_col] = df[review_col].copy()
     df[review_col] = df[review_col].apply(lambda x: x.lower())
     df[review_col] = df[review_col].apply(lambda x: remove_punctuation(x))
     df[review_col] = df[review_col].apply(lambda x: lemmatize_text(x))
     return df.drop_duplicates(subset=review_col) if drop_duplicates else df


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation + '’')
    return text.translate(translator)

def get_number_topics_subcategory(num_reviews):
     if num_reviews > 50000:
          return 20
     if num_reviews > 25000:
          return 15
     if num_reviews > 10000:
          return 10
     if num_reviews > 5000:
          return 8
     if num_reviews > 1000:
          return 6
     if num_reviews > 500:
          return 5
     if num_reviews > 100:
          return 4
     if num_reviews > 50:
          return 3
     if num_reviews > 10:
          return 2
     return 1

def calculate_num_topics_star_rating(total_topics, star_rating, min_topics=1):
     # logarithmic weighting (smooth it out)
     weight = math.log(6 - star_rating + 1)  # Adding 1 to avoid log(0)

     # normalize weight to ensure sum of the topics is not greater than total_topics
     total_weight = sum(math.log(6 - r + 1) for r in range(1, 6))
     normalized_weight = weight / total_weight

     # number of topics for the group based on normalized weight
     num_topics = max(min_topics, math.ceil(normalized_weight * total_topics))

     return num_topics

def generate_topic_label(top_words, rating, model='llama3-8b-8192'):
     client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
     system_message = """
     Generate a concise and coherent phrase that summarizes topics based on provided keywords
     and associated review ratings. The keywords are ordered by relevance from the most to the least.


     Ensure that each phrase accurately reflects the sentiment implied by the rating and highlights key aspects of the topic.
     For example, a rating of 1 should convey dissatisfaction or issues, while a rating of 5 should express satisfaction or positive feedback.
     Omit pronouns and conjunctions to keep the phrase succinct.


     For reviews with low ratings (1-2) or with negative keywords, focus on problematic aspects such as defects, damages, or poor quality and
     be as detailed as possible.


     Output only the phrase without additional commentary.


     Example:


     Input: "leak box, return, item, bottle leak, arrive damage, box, arrive, damage leak, damage, leak" Rating: 1
     Output: "Leaking or damaged items and boxes upon delivery"
     """
     user_message = f'Input: "{top_words}" rating: {rating}'
     response = client.chat.completions.create(
          messages=[
               {
                    "role": "system",
                    "content": system_message
               },
               {
                    "role": "user",
                    "content": user_message
               }
          ],
          model=model,
     )
     generated_phrase = response.choices[0].message.content
     try:
          if generated_phrase.startswith('Here'):
               generated_phrase = generated_phrase.split(':')[1]
     except:
          pass
     # remove double quotes
     generated_phrase = generated_phrase.replace('"', '')
     return generated_phrase

def similarity_scores(reviews, topics, model_name='all-MiniLM-L6-v2', chunk_size=1000):
     """
     Calculate the similarity scores between reviews and topics using a pre-trained SentenceTransformer model.
     Chunking is used to avoid memory issues when calculating similarity scores for a large number of reviews.

     model_name: the name of the pre-trained SentenceTransformer model to use
     reviews: a list of review texts
     topics: a list of topic phrases
     chunk_size: the number of reviews to process at a time

     return: a 2D numpy array of similarity scores
     """
     model = SentenceTransformer(model_name)
     num_reviews = len(reviews)
     similarity_scores = []
     total_time = 0
     for i in range(0, num_reviews, chunk_size):
          start = time.time()
          chunk_reviews = reviews[i:i + chunk_size]
          review_embeddings = model.encode(chunk_reviews, convert_to_tensor=True)
          chunk_topics = topics[i:i + chunk_size]
          phrase_embeddings = model.encode(chunk_topics, convert_to_tensor=True)
          chunk_similarity = cosine_similarity(review_embeddings, phrase_embeddings).cpu().numpy()
          similarity_scores.extend(chunk_similarity)
          time_i = time.time() - start
          print(f"Time to process chunk {i}-{i + chunk_size}: {time_i:.2f} seconds")
          total_time += time_i
     print(f"Total processing time: {total_time:.2f} seconds")
     return similarity_scores