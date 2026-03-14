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
    prediction = classifier.classify(features)

    dist = classifier.prob_classify(features)
    confidence = dist.prob(prediction)
    
    if prediction == "Opinion":
        sentiment = get_sentiment(user_input)
        print(f"Result: {prediction.upper()} ({sentiment.upper()}) ({confidence:.1%} confidence)")
    else:
        print(f"Result: {prediction.upper()} ({confidence:.1%} confidence)")
