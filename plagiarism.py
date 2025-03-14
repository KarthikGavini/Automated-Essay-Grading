# plagiarism_model.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
import joblib
import os
import re  # Add this line to import the 're' module
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define the path to the plagiarism_model folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), "plagiarism_model")

# Load precomputed data
def load_precomputed_data():
    preprocessed_texts = joblib.load(os.path.join(MODEL_DIR, "preprocessed_texts.joblib"))
    embeddings = joblib.load(os.path.join(MODEL_DIR, "embeddings.joblib"))
    nbrs = joblib.load(os.path.join(MODEL_DIR, "nearest_neighbors_model.joblib"))
    return preprocessed_texts, embeddings, nbrs

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return ' '.join(tokens)

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

# Plagiarism detection function
def check_plagiarism(user_essay):
    # Load precomputed data
    preprocessed_texts, embeddings, nbrs = load_precomputed_data()
    
    # Preprocess user essay
    user_embedding = get_bert_embedding(preprocess(user_essay))
    
    # Find closest match using ANN
    distances, indices = nbrs.kneighbors(user_embedding)
    closest_text = preprocessed_texts[indices[0][0]]
    closest_distance = distances[0][0]
    similarity_score = 1 - closest_distance
    
    # Interpret results
    if similarity_score > 0.8:
        risk_level = "High Plagiarism Risk"
    elif similarity_score > 0.5:
        risk_level = "Moderate Plagiarism Risk"
    else:
        risk_level = "Low Plagiarism Risk"
    
    return similarity_score * 100, risk_level, closest_text
