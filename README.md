# 📋 BOAMP Procurement Notice Classifier

> Automated NLP pipeline to classify French public procurement notices (BOAMP) as relevant or irrelevant for IGN's national geographic vector database — dramatically reducing manual review workload.

---

## 📌 Project Context

This project was developed as part of a university collaboration between **Université Lyon 2** and the **Institut National de l'Information Géographique et Forestière (IGN)**.

The **BOAMP** (*Bulletin Officiel des Annonces des Marchés Publics*) is the French official journal for public procurement notices. The IGN regularly monitors these notices to detect any construction or infrastructure projects that may require an update to their national geographic vector database.

The goal of this project is to **automate** that monitoring process by building a machine learning classifier that predicts whether a given BOAMP notice should be flagged as:

- ✅ **"Pris en compte"** — Relevant: the notice may impact IGN's geographic data
- ❌ **"Rejeté (hors specs)"** — Irrelevant: the notice does not concern the IGN

The dataset contains approximately **6,000 labeled BOAMP notices**.

---

## 🗂️ Project Structure

```
boamp-procurement-classifier/
│
├── classification_boamp_project.ipynb   # Main notebook (full pipeline)
├── train.jsonl                          # Training data (not included — see Data section)
├── test.jsonl                           # Test data (not included — see Data section)
└── README.md
```

---

## ⚙️ Pipeline Overview

```
Raw BOAMP notices (text)
        │
        ▼
  Text Preprocessing
  (cleaning, lowercasing, stopword removal, accent normalization)
        │
        ▼
    Lemmatization
  (spaCy fr_core_news_sm)
        │
        ▼
  Feature Extraction
  (CountVectorizer / TF-IDF)
        │
        ▼
  Dimensionality Reduction
  (SVD / TruncatedSVD)
        │
        ▼
  Classification Models
  ├── Logistic Regression  ✅ Best model
  ├── Decision Tree
  ├── Naive Bayes (MultinomialNB)
  └── Random Forest (with hyperparameter tuning)
        │
        ▼
  Evaluation
  (Accuracy, F1-score, Confusion Matrix)
```

---

## 🔬 Methods & Techniques

### Text Preprocessing
- Lowercasing and accent normalization (`unidecode`)
- Punctuation and special character removal
- French stopword removal (`NLTK`)

### Lemmatization
Lemmatization was chosen over stemming for its ability to preserve linguistic meaning — particularly important in the formal, technical language of administrative procurement notices. The French spaCy model `fr_core_news_sm` was used.

### Vectorization
- **CountVectorizer** (Bag of Words)
- **TF-IDF Vectorizer** — captures term importance relative to the corpus

### Dimensionality Reduction
- **Truncated SVD** (LSA) applied to reduce the feature space of sparse TF-IDF matrices before feeding into classifiers

### Models Compared
| Model | Notes |
|---|---|
| Logistic Regression | Best overall — handles sparse text data well, interpretable |
| Decision Tree | Fast but prone to overfitting |
| Multinomial Naive Bayes | Efficient baseline for text classification |
| Random Forest | Robust, tuned via RandomizedSearchCV |

### Hyperparameter Tuning
Random Forest was tuned using **RandomizedSearchCV** with cross-validation across parameters including `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `bootstrap`.

---

## 🏆 Results

**Logistic Regression** emerged as the best-performing model for this task due to:
- Excellent handling of high-dimensional sparse text matrices
- Strong generalization with limited overfitting risk
- Interpretable coefficients — useful for understanding which terms drive classification decisions
- Good balance between performance and computational efficiency for production use

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy unidecode
python -m spacy download fr_core_news_sm
```

**Main libraries:**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — vectorization, modeling, evaluation
- `spaCy` (`fr_core_news_sm`) — French lemmatization
- `NLTK` — French stopwords
- `unidecode` — accent normalization
- `matplotlib`, `seaborn` — visualization

---

## 📊 Data

The dataset consists of BOAMP notices in `.jsonl` format, labeled with a binary target variable `cal_réponse_signalement`:

| Label | Meaning |
|---|---|
| `Pris en compte` | Notice is relevant to IGN's geographic database |
| `Rejeté (hors specs)` | Notice is out of scope for IGN |

> ⚠️ The raw data is not included in this repository as it is proprietary to the IGN project.

---

## 👤 Author

**Maxence Chan**  
📧 maxencechan@orange.fr  
🔗 [LinkedIn](https://linkedin.com/in/maxence-chan)

---

## 📝 License

This project was developed for academic purposes in collaboration with IGN. Please contact the author for any reuse or adaptation.
