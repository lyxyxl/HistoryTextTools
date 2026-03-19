import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

def extract_features(document):
    document_words = set(word_tokenize(document.lower()))
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

def get_sentiment(text):
    scores = sia.polarity_scores(str(text))

    if scores['compound'] > 0.05:
        return "Positive"
    elif scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral Opinion"

# Load CSV file
try:
    with open('src/history_bias_model.pickle', 'rb') as f:
        classifier = pickle.load(f)
    with open('src/model_vocabulary.pickle', 'rb') as f:
        word_features = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Did you run sentiment_model_trainer.py?")
    exit()

# Initialise VADER
sia = SentimentIntensityAnalyzer()

print("--- History Bias Detector (Type 'quit' to exit) ---")
print("Enter a sentence to check if it's Fact or Opinion.")

# Testing Loop
while True:
    user_input = input("\n> ")
    if user_input.lower() == 'quit':
        break
    
    features = extract_features(user_input)
    # 1. Get the probability distribution instead of just the label
    dist = classifier.prob_classify(features)
    prediction = dist.max()
    confidence = dist.prob(prediction)
    
    # 2. Apply the Threshold Logic
    # If the model thinks it's an Opinion but isn't very sure (e.g., < 80%), 
    # we treat it as a factual/Neutral statement for safety.
    if prediction == "Opinion" and confidence < 0.80:
        final_prediction = "Neutral"
    else:
        final_prediction = prediction

    # 3. Print results based on the FINAL decision
    if final_prediction == "Opinion":
        sentiment = get_sentiment(user_input)
        print(f"Result: {final_prediction.upper()} ({sentiment.upper()}) ({confidence:.1%} confidence)")
    else:
        # Note: If we overrode the label, we still show the original confidence 
        # to help you understand why the filter triggered.
        print(f"Result: {final_prediction.upper()} ({confidence:.1%} confidence)")