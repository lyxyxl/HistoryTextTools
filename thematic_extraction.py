import gensim
from gensim import corpora
from gensim.models import CoherenceModel, TfidfModel
import spacy
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import re

CSV_FILE = "src/themes_data.csv"  
TEXT_COLUMN = "Text"

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords', quiet=True)

def preprocess_text(docs):
    # 1. Expand stop words to include "leaky" words that appear everywhere
    stop_words = set(stopwords.words('english'))
    stop_words.update([
        'hitler', 'nazi', 'german', 'germany', 'also', 'could', 'would', 
        'power', 'people', 'party', 'able', 'many', 'set', 'make', 'use',
        'result', 'impact', 'take', 'way', 'become', 'even', 'one', 'nazi',
        'nazis'
    ])
    
    processed = []
    for doc in nlp.pipe(docs, disable=["ner", "parser"]):
        # Remove weird encoding artifacts like 'â' or '\x9d'
        text_clean = re.sub(r'[^\x00-\x7f]', r'', doc.text) 
        
        # POS filtering: Focus on Nouns and Adjectives
        clean_tokens = [
            token.lemma_.lower() for token in nlp(text_clean) 
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]
            and token.lemma_.lower() not in stop_words 
            and token.text not in string.punctuation 
            and len(token.text) > 2 
        ]
        processed.append(clean_tokens)
    
    # 2. Add Bigrams
    bigram = gensim.models.Phrases(processed, min_count=2, threshold=15)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in processed]

def run_topic_model(processed, num_topics):
    id2word = corpora.Dictionary(processed)
    
    # 3. Aggressive filtering: Ignore words in > 50% of documents
    id2word.filter_extremes(no_below=2, no_above=0.5) 
    
    corpus = [id2word.doc2bow(text) for text in processed]
    
    # tfidf = TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics, 
        random_state=1000,
        passes=150,          # More passes for a small dataset
        iterations=1000,
        alpha='auto', 
        eta=0.5
    )
    return lda_model, corpus, id2word

# (Remaining print_topics and evaluate_model functions stay the same)

if __name__ == "__main__":
    # Load with encoding='latin1' to avoid character errors
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    print(f"Loaded {len(df)} documents. Preprocessing...")
    
    clean_docs = preprocess_text(df[TEXT_COLUMN].astype(str).tolist())
    
    num_topics = 5
    lda_model, corpus, id2word = run_topic_model(clean_docs, num_topics)

    print("\n--- Refined Themes ---")

    for idx, topic in lda_model.show_topics(num_topics=-1, formatted=False):
        words = [word.replace("_", " ") for word, prob in topic]  
        
        print(f"\nTopic {idx}")
        print(f"Keywords: {', '.join(words[:6])}")
        
    coherence_model_lda = CoherenceModel(model=lda_model, texts=clean_docs, dictionary=id2word, coherence='c_v')
    print(f'\nCoherence Score: {coherence_model_lda.get_coherence():.4f}')