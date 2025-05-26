import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
import joblib
import os

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load and train the model once
print("Loading data...")
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'cleaned_terrorism_nlp_ready.csv'))

df = df.drop(columns=['User_Category'])
# Clean the text data
print("Cleaning text data...")
X = df['Post_Text'].apply(clean_text)
y = df['Sentiment']

# Optimize CountVectorizer parameters
print("Creating features...")
vectorizer = CountVectorizer(
    max_features=10000,      # Increase vocabulary size
    min_df=2,               # Remove rare words
    max_df=0.95,           # Remove very common words
    ngram_range=(1, 2),    # Use both single words and pairs of words
    strip_accents='unicode',
    binary=True            # Use binary features instead of counts
)
X_vec = vectorizer.fit_transform(X)

# Optimize LogisticRegression parameters
print("Training model...")
model = LogisticRegression(
    C=1.0,                # Inverse of regularization strength
    class_weight='balanced',  # Handle imbalanced classes
    max_iter=1000,        # Increase max iterations
    random_state=42,      # For reproducibility
    solver='liblinear',   # Better for small datasets
    penalty='l1'          # L1 regularization for feature selection
)
model.fit(X_vec, y)

# Save models in the current directory
print("Saving models...")
current_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(vectorizer, os.path.join(current_dir, 'vectorizer.pkl'))
joblib.dump(model, os.path.join(current_dir, 'model.pkl'))
print("Model training completed!")
