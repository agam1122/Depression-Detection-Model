import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# --- CONFIGURATION ---
MODEL_FILE = 'DD_model_lr.pkl'
VECTORIZER_FILES = ['X_train_tfidf.pkl', 'X_test_tfidf.pkl', 'y_train.pkl', 'y_test.pkl']
DATA_FILE = "data.csv"


def clean_text(text):
    """
    Standardizes text by removing URLs, mentions, and special characters.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text


if all(os.path.exists(f) for f in VECTORIZER_FILES):
    X_train_tfidf = joblib.load('X_train_tfidf.pkl')
    X_test_tfidf = joblib.load('X_test_tfidf.pkl')
    y_train = joblib.load('y_train.pkl')
    y_test = joblib.load('y_test.pkl')
    
else:
    df = pd.read_csv("data.csv")

    df.drop("Unnamed: 0", axis=1, inplace=True)

    target_values = ['teenagers', 'depression', 'SuicideWatch', 'happy', 'DeepThoughts']
    df = df.query('subreddit in @target_values')

    df["text"] = df["title"] + ". " + df["body"]

    df = df.dropna()

    labels = df["label"]

    features = df.drop(["label", "subreddit"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        labels, 
        stratify=df['subreddit'],
        test_size=0.1, 
        random_state=42
    )

    X_train_cleaned = X_train['text'].apply(clean_text)
    X_test_cleaned = X_test['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_cleaned)
    X_test_tfidf = vectorizer.transform(X_test_cleaned)

    # Save the vectorizers 
    joblib.dump(X_train_tfidf, 'X_train_tfidf.pkl')
    joblib.dump(X_test_tfidf, 'X_test_tfidf.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(y_test, 'y_test.pkl')
    joblib.dump(vectorizer, 'Vectorizer.pkl')

    print("Vectorizers saved successfully!")


if os.path.exists(MODEL_FILE):
    lr_model = joblib.load(MODEL_FILE)
else:
    
    # # 1. Initialize the model
    # nb_model = MultinomialNB()
    
    # # 2. Train the model using the TF-IDF matrix and your labels
    # # Make sure y_train is the correct length 
    # nb_model.fit(X_train_tfidf, y_train)

    # 1. Initialize with class_weight='balanced'
    # This tells the model to pay extra attention to the minority class (Class 1.0)
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')

    # 2. Train (This will take a few seconds longer than NB)
    lr_model.fit(X_train_tfidf, y_train)

    # rf_model = RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=100,
    #     random_state=42,
    #     n_jobs=-1  # Uses all available CPU cores for faster training
    # )
    #
    # rf_model.fit(X_train_tfidf, y_train)
    #
    # Save the trained model
    joblib.dump(lr_model, MODEL_FILE)
    
    print("Model saved successfully!")


# # 1. Make predictions
# y_pred = nb_model.predict(X_test_tfidf)

# # 2. Check the results
# print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
# print("\nDetailed Report:")
# print(classification_report(y_test, y_pred))

# 3. Predict and Compare
y_pred_rf = lr_model.predict(X_test_tfidf)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")
print(classification_report(y_test, y_pred_rf))

























