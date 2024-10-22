# # from django.shortcuts import render

# # Create your views here.
# # faqs/views.py
# import json
# from django.shortcuts import render
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import numpy as np
# import os

# # Download NLTK stopwords and tokenizer
# nltk.download('stopwords')
# nltk.download('punkt')

# # Path for FAQ data
# # def load_faq_data():
# #     json_file_path = os.path.join(os.path.dirname(__file__), 'faqs.json')
# #     with open(json_file_path, 'r') as f:
# #         faq_data = json.load(f)
# #     return faq_data

# def load_faq_data(file_path):
#     with open(file_path, 'r') as f:  # Changed mode to 'r' for reading
#         faq_data = json.load(f)
#     return faq_data
# faq_data = load_faq_data('C:\Users\ayush\faq_final\faq_final\asgi.py')
# # Preprocess text
# def preprocess(text):
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(text.lower())
#     filtered_words = [w for w in word_tokens if not w in stop_words and w.isalnum()]
#     return " ".join(filtered_words)

# # Search FAQs
# def find_relevant_faq(query, faq_data):
#     all_faqs = []
#     for category, faqs in faq_data.items():
#         all_faqs.extend(faqs)
#     questions = [faq['question'] for faq in all_faqs]
#     answers = [faq['answer'] for faq in all_faqs]

#     processed_questions = [preprocess(question) for question in questions]
#     processed_query = preprocess(query)
import json
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from django.conf import settings
import os

nltk.download('stopwords')
nltk.download('punkt')

# Load the JSON file
def load_faq_data(file_path):
    with open(file_path, 'r') as f:  # Changed mode to 'r' for reading
         faq_data = json.load(f)
    return faq_data
    # Preprocess text
faq_data = load_faq_data('C:\Users\ayush\faq_final\faq_final\asgi.py')   
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered_words)

# Search FAQs
def find_relevant_faq(query, faq_data):
    all_faqs = []
    for category, faqs in faq_data.items():
        all_faqs.extend(faqs)

    questions = [faq['question'] for faq in all_faqs]
    answers = [faq['answer'] for faq in all_faqs]

    processed_questions = [preprocess(question) for question in questions]
    processed_query = preprocess(query)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_questions + [processed_query])

    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])
    top_match_idx = np.argmax(similarity_scores)

    return questions[top_match_idx], answers[top_match_idx]

# View function for FAQ page
def faq_view(request):
    question, answer = None, None
    faq_data = load_faq_data()

    if request.method == 'POST':
        query = request.POST.get('search_query', '')
        if query:
            question, answer = find_relevant_faq(query, faq_data)

    return render(request, 'faq/faq_page.html', {'question': question, 'answer': answer})
