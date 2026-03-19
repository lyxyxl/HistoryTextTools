import pandas as pd
import nltk
import pickle
import random
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics import precision, recall, f_measure
from collections import defaultdict
from nltk.util import ngrams

# Setup
CSV_FILE = "src/sentiment_data.csv"  
TEXT_COLUMN = "Text"
LABEL_COLUMN = "Label"

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words('english'))
extra_stops = {
    'also', 'part', 'sent', 'june', 'ii', 'time', 'year',
    'two', 'years', 'towards'
}
stop_words.update(extra_stops)

def extract_features(document):
    words = word_tokenize(str(document).lower())
    document_words = [w for w in words if w.isalpha()]

    text = " ".join(document_words)

    features = {}

    # Existing word features
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)

    # 🔥 NEW: Sentence structure features

    # Definition-style (Neutral)
    features['is_definition'] = any(
        text.startswith(pattern) for pattern in [
            "the", "a", "an"
        ]
    ) and any(
        word in document_words for word in ["is", "are", "was", "were", "has"]
    )

    # Passive / factual tone (Neutral)
    features['has_passive'] = any(
        word in document_words for word in ["was", "were", "has", "had"]
    )

    # Numbers → often factual
    features['has_number'] = any(char.isdigit() for char in document)

    # Opinion indicators
    features['has_opinion_word'] = any(
        word in document_words for word in [
            "should", "must", "believe", "think", "argue", "suggest"
        ]
    )

    # Contrast words → often opinion/analysis
    features['has_contrast'] = any(
        word in document_words for word in [
            "however", "although", "but", "instead"
        ]
    )

    # Neutral tone indicators
    features['is_descriptive'] = any(
        word in document_words for word in [
            "is", "are", "was", "were", "has", "have"
        ]
    )

    features['has_superlative'] = any(
        word in document_words for word in [
            "largest", "smallest", "highest", "lowest", "first", "last"
        ]
    )

    features['has_proper_noun'] = any(
        word.istitle() for word in document.split()
    )

    features['has_past_tense'] = any(
        word.endswith("ed") for word in document_words
    )

    return features

# Load CSV file
print("Loading dataset...")
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
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
    clean_words = [
        w for w in words 
        if w.isalpha() 
        and len(w) > 2 
        and w not in stop_words
    ]

    all_words.extend(clean_words)

all_words_freq = nltk.FreqDist(all_words)
# Reduce noise
word_features = list(all_words_freq.keys())[:1200]

print("Building bigram features...")

bigram_list = []
for text, label in data:
    words = word_tokenize(str(text).lower())
    clean_words = [
        w for w in words 
        if w.isalpha() 
        and len(w) > 2
    ]
    bigram_list.extend(["_".join(bg) for bg in ngrams(clean_words, 2)])

bigram_freq = nltk.FreqDist(bigram_list)
bigram_features = list(bigram_freq.keys())[:200]  # limit size

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



print("\nCalculating precision, recall, and F1-score...")

# Create reference and predicted sets
refsets = defaultdict(set)
testsets = defaultdict(set)

for i, (features, label) in enumerate(test_set):
    refsets[label].add(i)
    predicted = classifier.classify(features)
    testsets[predicted].add(i)

# Get all labels
labels = set(label for (_, label) in test_set)

# Calculate metrics per label
for label in labels:
    p = precision(refsets[label], testsets[label])
    r = recall(refsets[label], testsets[label])
    f1 = f_measure(refsets[label], testsets[label])

    print(f"\nLabel: {label}")
    print(f"  Precision: {p:.3f}" if p else "  Precision: N/A")
    print(f"  Recall:    {r:.3f}" if r else "  Recall: N/A")
    print(f"  F1-Score:  {f1:.3f}" if f1 else "  F1-Score: N/A")

f1_scores = []

for label in labels:
    f1 = f_measure(refsets[label], testsets[label])
    if f1:
        f1_scores.append(f1)

macro_f1 = sum(f1_scores) / len(f1_scores)
print(f"\nOverall Macro F1 Score: {macro_f1:.3f}")


print("\nMost Informative Features (What the AI learned):")
classifier.show_most_informative_features(10)

# Save model
print("\nSaving model to file...")

with open('src/history_bias_model.pickle', 'wb') as f:
    pickle.dump(classifier, f)

with open('src/model_vocabulary.pickle', 'wb') as f:
    pickle.dump(word_features, f)

print("Done! You can now use 'sentiment_model'.")
