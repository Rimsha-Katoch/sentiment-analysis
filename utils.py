import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load your vectorizer and model
vectorizer = joblib.load(os.path.join(current_dir, 'vectorizer.pkl'))
model = joblib.load(os.path.join(current_dir, 'model.pkl'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_label(text):
    try:
        # Clean the input text
        cleaned_text = clean_text(text)
        
        # Transform the text into vector format
        text_vec = vectorizer.transform([cleaned_text])

        # Get prediction and probability
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        
        # Get the appropriate probability based on the prediction
        if prediction == "negative":
            label = "Suspect"
            confidence = probabilities[0]  # Probability of negative class
        else:
            label = "Non-Suspect"
            confidence = probabilities[1]  # Probability of positive class

        # Scale confidence to range 60-86
        scaled_confidence = 60 + (confidence * 26)  # This will map 0-1 to 60-86
        scaled_confidence = min(scaled_confidence, 86.0)  # Ensure it doesn't exceed 86
        
        return label, scaled_confidence
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error", 0.0
