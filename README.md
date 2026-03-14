URECA Research Project

This project provides a pipeline for analyzing historical text data (exam papers, notes, and archives) related to Nazi Germany. It uses Latent Dirichlet Allocation (LDA) for thematic extraction and a Naive Bayes Classifier to identify linguistic biases or classifications.

## Project Structure

project_root/
│
├── src/                        # Saved model files (.pickle, .model, .dict)
├── data/                       # Subfolder for raw CSV data
├── thematic_trainer.py         # Script to train the LDA Topic Model
├── sentiment_trainer.py        # Script to train the Naive Bayes Classifier
├── search_tool.py              # Multi-document fuzzy search & retrieval engine
├── themes_data.csv             # Dataset for topic modeling
└── sentiment_data.csv          # Dataset for sentiment/bias training

## Features
Multi-Format Support: Extracts text from .docx and .pdf files.

Thematic Modeling: Uses Gensim's LDA with TF-IDF re-weighting to identify high-level historical themes.

Bias Classification: A trained Naive Bayes model to categorize text based on frequent word features.

Smart Retrieval: Fuzzy search logic that handles spelling variations (e.g., "Nuremburg" vs "Nuremberg") and extracts key dates.


## Installation & Setup

Install required dependencies:
pip install pandas nltk gensim spacy pymupdf python-docx
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

Usage
1. Training the Thematic Model
Run this to generate topics from your history corpus.
python thematic_trainer.py

3. Training the Sentiment/Bias Model
This script processes sentiment_data.csv and saves the resulting classifier into the src/ folder.

python sentiment_trainer.py
