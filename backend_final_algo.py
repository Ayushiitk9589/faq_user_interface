# pip install scikit-learn nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np


# Load the JSON file
def load_faq_data(file_path):
    with open(file_path, 'r') as f:  # Changed mode to 'r' for reading
        faq_data = json.load(f)
    return faq_data


# Preprocess text
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [w for w in word_tokens if not w in stop_words and w.isalnum()]
    return " ".join(filtered_words)


# Search FAQs
def find_relevant_faq(query, faq_data):
    # Iterate through all FAQs in all categories
    all_faqs = []
    for category, faqs in faq_data.items():  # Get both category and FAQs
        all_faqs.extend(faqs)  # Add FAQs to the list

    # Extract questions and answers
    questions = [faq['question'] for faq in all_faqs]  # Access question for each FAQ
    answers = [faq['answer'] for faq in all_faqs]  # Access answer for each FAQ

    processed_questions = [preprocess(question) for question in questions]
    processed_query = preprocess(query)

    # Use TF-IDF and cosine similarity to find the most relevant FAQ
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_questions + [processed_query])

    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])
    top_match_idx = np.argmax(similarity_scores)

    return questions[top_match_idx], answers[top_match_idx]


# Load the FAQ data outside the function, using the provided file path
faq_data = load_faq_data('/content/faqs.json')

query = input("Enter your question: ")
print(find_relevant_faq(query, faq_data))