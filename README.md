# HistoryTextTools: NTU URECA Research Project

An advanced Natural Language Processing (NLP) pipeline designed to analyze historical examination papers, extract long-term thematic trends, detect latent biases, and enable robust semantic retrieval across multi-format academic documents. 

This repository houses the core text-mining tools developed under the **Undergraduate Research Experience on CAmpus (URECA)** program.

---

## 🚀 Project Overview

Analyzing qualitative historical data at scale presents unique challenges due to varying linguistic architectures, multi-format text sources, and the necessity for granular semantic precision. This project addresses these hurdles by building a robust Python-based NLP pipeline tailored for historical and educational research. 

By employing specialized linguistic heuristics and advanced machine learning models, the system converts unstructured historical exam papers into structured, quantifiable insights—enhancing academic rigor and workflow reproducibility.

### Core Features & Models
The toolkit is divided into three specialized, standalone NLP components:
1. **Thematic Extraction Engine (Topic Modeling):** Built on Latent Dirichlet Allocation (LDA) via `Gensim` to discover recurring themes, shifts in educational focus, and conceptual evolution over historical timelines.
2. **Sentiment & Bias Analysis Module:** Utilizes custom linguistic heuristics alongside `NLTK` and `spaCy` to map underlying sentiments, tones, and institutional biases embedded within historical questioning.
3. **Fuzzy Semantic Search Engine:** A cross-document information retrieval system that matches concepts rather than just exact keywords, ensuring highly relevant search capability across diverse document formats.

---

## 🛠️ Tech Stack & Dependencies

The pipeline is built purely in Python and relies on industry-standard libraries for linguistic engineering, text processing, and statistical modeling:

* **Core NLP Frameworks:** `spaCy` (advanced tokenization, POS tagging, named entity recognition), `NLTK` (text cleaning, stop-word filtering, sentiment scoring).
* **Topic Modeling:** `Gensim` (dictionary mapping, LDA model building, coherence scoring).
* **Data Structures & Analytics:** `Pandas`, `NumPy` (high-performance tabular data extraction and matrix operations).

---

## 📁 Repository Structure

```text
├── data/                   # Input directory for raw historical texts & exam papers
├── src/                    # Source code for the NLP pipeline
│   ├── preprocessing.py    # Custom linguistic cleaning, lemmatization, & tokenization
│   ├── topic_modeling.py   # Thematic Extraction Engine (LDA)
│   ├── sentiment_analysis.py # Tone, sentiment, and bias analysis module
│   └── semantic_search.py  # Fuzzy semantic search and retrieval system
├── notebooks/              # Jupyter Notebooks for exploratory data analysis (EDA)
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

Ensure you have Python 3.10+ installed on your system.

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/lyxyxl/HistoryTextTools.git](https://github.com/lyxyxl/HistoryTextTools.git)
   cd HistoryTextTools

2. **Create and Activate a Virtual Environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Download Required Language Models:**
   The pipeline requires specific linguistic corpora from `spaCy` and `NLTK`. Download them by running:
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"

---

## 📖 Usage Guide

1. **Data Preparation**
   Place your raw text documents or structured text inputs into the `data/` folder. The pipeline expects unformatted or semi-formatted historical examination logs.

2. **Preprocessing Data**
  Before running any analysis, clean and parse the historical text:
  ```bash
  python src/preprocessing.py --input data/raw_exams.csv --output data/cleaned_exams.pkl
  ```

3. **Training the Thematic Extraction Engine (LDA)**
   To extract key historical themes and optimize them using topic coherence scores:
   ```bash
   python src/topic_modeling.py --num_topics 5 --passes 20

4. **Running Sentiment & Bias Detection**
   Running Sentiment & Bias Detection
   ```bash
   python src/sentiment_analysis.py --input data/cleaned_exams.pkl

5. **Executing Semantic Fuzzy Queries**
   To search across multi-format documents using the conceptual/semantic lookup tool:
   ```bash
   python src/semantic_search.py --query "colonial policy evaluation"

---
