import pandas as pd
import nltk
import pickle
import random
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Setup
CSV_FILE = "src/sentiment_data.csv"  
TEXT_COLUMN = "Text"
LABEL_COLUMN = "Label"

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
extra_stops = {'also', 'part', 'sent', 'june', 'ii', 'time', 'year', 'two', 'years', 'towards'}
stop_words.update(extra_stops)

def extract_features(document):
    words = word_tokenize(str(document).lower())
    document_words = set([w for w in words if w not in stop_words and w.isalpha()])

    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Load CSV file
print("Loading dataset...")
try:
    df = pd.read_csv(CSV_FILE, encoding='cp1252')
    df.drop_duplicates(inplace=True)
    # Filter: We only need Text and Label for training
    # We deliberately drop 'Topic' so the AI focuses on language, not subjects
    data = list(zip(df[TEXT_COLUMN],df[LABEL_COLUMN]))
    print(f"Loaded {len(data)} rows.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Build vocab list of top 3000 most frequent words
print("Building vocabulary...")
all_words = []
for text, label in data:
    words = word_tokenize(str(text).lower())
    clean_words = [w for w in words if w not in stop_words and w.isalpha()]

    all_words.extend(clean_words)

all_words_freq = nltk.FreqDist(all_words)
word_features = list(all_words_freq.keys())[:3000]


# Clean up text
print("Processing features (this may take a moment)...")
featuresets = [(extract_features(text), label) for (text, label) in data]
random.shuffle(featuresets)
cutoff = int(len(featuresets) * 0.8)
train_set = featuresets[:cutoff]
test_set = featuresets[cutoff:]

# Train model
print(f"Training on {len(train_set)} examples...")
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Results
print("Calculating accuracy...")
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"\nModel Accuracy: {accuracy:.1%}")

print("\nMost Informative Features (What the AI learned):")
classifier.show_most_informative_features(10)

# Save model
print("\nSaving model to file...")

with open('src/history_bias_model.pickle', 'wb') as f:
    pickle.dump(classifier, f)

with open('src/model_vocabulary.pickle', 'wb') as f:
    pickle.dump(word_features, f)

print("Done! You can now use 'sentiment_model'.")