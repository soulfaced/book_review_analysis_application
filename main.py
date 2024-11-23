# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
import xgboost as xgb
import os
import nltk

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Step 1: Load Data
print("Loading data...")
books_reviews = pd.read_csv("Books_rating.csv").head(10000)  # Limit data size for faster processing
books_details = pd.read_csv("books_data.csv").head(10000)
print("Data loaded successfully.")

# Step 2: Merge Datasets
print("Merging datasets on Title...")
data = pd.merge(books_reviews, books_details, on="Title", how="inner")
print(f"Merged dataset contains {len(data)} rows and {len(data.columns)} columns.")

# Step 3: Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in stop_words])
    return text

data['review/text'] = data['review/text'].fillna("").apply(preprocess_text)

# Step 4: Feature Engineering
print("Feature Engineering...")

def calculate_helpfulness_ratio(value):
    try:
        if isinstance(value, str) and '/' in value:
            numerator, denominator = map(int, value.split('/'))
            return numerator / max(denominator, 1)
        else:
            return 0
    except:
        return 0

# Individual Review Features
data['helpfulness_ratio'] = data['review/helpfulness'].apply(calculate_helpfulness_ratio)
data['word_count'] = data['review/text'].apply(lambda x: len(x.split()))
positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'awesome'}
negative_words = {'bad', 'poor', 'terrible', 'worst', 'hate'}
data['positive_words'] = data['review/text'].apply(lambda x: len([word for word in x.split() if word in positive_words]))
data['negative_words'] = data['review/text'].apply(lambda x: len([word for word in x.split() if word in negative_words]))

# Sentiment Analysis Features
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

data[['sentiment_polarity', 'sentiment_subjectivity']] = data['review/text'].apply(lambda x: pd.Series(get_sentiment(x)))

# Aggregate Features for Each Book
book_stats = data.groupby('Title').agg(
    avg_helpfulness_ratio=('helpfulness_ratio', 'mean'),
    avg_rating=('review/score', 'mean'),
    total_reviews=('review/text', 'count'),
    avg_word_count=('word_count', 'mean'),
    avg_sentiment_polarity=('sentiment_polarity', 'mean'),
    avg_sentiment_subjectivity=('sentiment_subjectivity', 'mean')
).reset_index()

# Merge Aggregate Features Back to Reviews
data = data.merge(book_stats, on='Title', how='left')

# Simulated Labels
print("Creating simulated 'Label' column...")
data['Label'] = data.apply(
    lambda x: 1 if x['helpfulness_ratio'] < 0.5 and x['review/score'] in [1, 5] else 0, axis=1
)

# Step 5: Split Data
X_numeric = data[[
    'helpfulness_ratio', 'word_count', 'positive_words', 'negative_words',
    'sentiment_polarity', 'sentiment_subjectivity',
    'avg_helpfulness_ratio', 'avg_rating', 'total_reviews', 'avg_word_count',
    'avg_sentiment_polarity', 'avg_sentiment_subjectivity'
]]
y = data['Label']
X_numeric_train, X_numeric_val, y_train, y_val = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Step 6: Text Vectorization
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_text_sparse = tfidf.fit_transform(data['review/text'])
X_text_train = X_text_sparse[:X_numeric_train.shape[0]]
X_text_val = X_text_sparse[X_numeric_train.shape[0]:]

# Combine Features
X_train_combined = hstack([X_numeric_train, X_text_train])
X_val_combined = hstack([X_numeric_val, X_text_val])

# Step 7: Model Training
print("Training XGBoost model...")
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_clf.fit(X_train_combined, y_train)

# Step 8: Validation
y_pred = xgb_clf.predict(X_val_combined)
print("Validation Results:")
print(classification_report(y_val, y_pred))
print(f"Accuracy: {accuracy_score(y_val, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.2f}")

# Step 9: Save Model and Vectorizer
os.makedirs("./models", exist_ok=True)
with open("./models/xgb_review_checker_model.pkl", "wb") as f:
    pickle.dump(xgb_clf, f)

with open("./models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Step 10: Predict User Review
def predict_review(new_data):
    with open("./models/xgb_review_checker_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("./models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # Process New Data
    new_data['review/text'] = new_data['review/text'].apply(preprocess_text)
    new_data['helpfulness_ratio'] = new_data['review/helpfulness'].apply(calculate_helpfulness_ratio)
    new_data['word_count'] = new_data['review/text'].apply(lambda x: len(x.split()))
    new_data['positive_words'] = new_data['review/text'].apply(lambda x: len([word for word in x.split() if word in positive_words]))
    new_data['negative_words'] = new_data['review/text'].apply(lambda x: len([word for word in x.split() if word in negative_words]))
    new_data[['sentiment_polarity', 'sentiment_subjectivity']] = new_data['review/text'].apply(lambda x: pd.Series(get_sentiment(x)))

    # Add Aggregate Features for the Book
    new_data = new_data.merge(book_stats, on='Title', how='left')

    # Extract Features
    text_features = tfidf.transform(new_data['review/text'])
    numeric_features = new_data[[
        'helpfulness_ratio', 'word_count', 'positive_words', 'negative_words',
        'sentiment_polarity', 'sentiment_subjectivity',
        'avg_helpfulness_ratio', 'avg_rating', 'total_reviews', 'avg_word_count',
        'avg_sentiment_polarity', 'avg_sentiment_subjectivity'
    ]].values

    features = hstack([numeric_features, text_features])
    predictions = model.predict(features)
    return ["Real" if pred == 1 else "Fake" for pred in predictions]

## Example Test Data
new_test_data = pd.DataFrame([
    {
        "Title": "Book A",
        "review/helpfulness": "2/3",  # Moderate helpfulness
        "review/text": "This is a great book. I loved it!",  # Positive review
        "review/score": 5  # High rating
    },
    {
        "Title": "Book A",
        "review/helpfulness": "1/10",  # Very low helpfulness
        "review/text": "Terrible book. Waste of time.",  # Negative review
        "review/score": 1  # Very low rating
    },
    {
        "Title": "Book B",
        "review/helpfulness": "10/10",  # High helpfulness
        "review/text": "Amazing book! Highly recommend.",  # Positive review
        "review/score": 5  # High rating
    },
    {
        "Title": "Book C",
        "review/helpfulness": "0/0",  # No helpfulness info
        "review/text": "Not worth it. The content is outdated.",  # Negative review
        "review/score": 2  # Low rating
    },
    {
        "Title": "Book D",
        "review/helpfulness": "5/5",  # High helpfulness
        "review/text": "The book is good, but the delivery was slow.",  # Neutral review
        "review/score": 4  # Decent rating
    },
    {
        "Title": "Book E",
        "review/helpfulness": "1/1",  # High helpfulness
        "review/text": "I bought this book as a gift, and it exceeded expectations.",  # Positive review
        "review/score": 5  # High rating
    },
    {
        "Title": "Book F",
        "review/helpfulness": "3/4",  # Moderate helpfulness
        "review/text": "This book is mediocre at best. Not worth the price.",  # Negative review
        "review/score": 3  # Average rating
    }
])

# Predict
predictions = predict_review(new_test_data)
for review, prediction in zip(new_test_data["review/text"], predictions):
    print(f"Review: {review} --> Prediction: {prediction}")
