import math
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

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

def generate_topic_label(client, model, top_words, rating):
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
   return generated_phrase


def similarity_scores(model_name, reviews, topics):
   """
   Calculate the similarity scores between reviews and topics using a pre-trained SentenceTransformer model.
  
   model_name: the name of the pre-trained SentenceTransformer model to use
   reviews: a list of review texts
   topics: a list of topic phrases
  
   return: a 2D numpy array of similarity scores
   """
   model = SentenceTransformer(model_name)
   review_embeddings = model.encode(reviews, convert_to_tensor=True)
   phrase_embeddings = model.encode(topics, convert_to_tensor=True)
   similarity_scores = cosine_similarity(review_embeddings, phrase_embeddings).cpu().numpy()
   return similarity_scores

