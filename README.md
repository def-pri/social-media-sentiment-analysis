**Social Media Sentiment Analysis — 3.1 Million Tweets (NLP + Machine Learning)**

A full data science pipeline analyzing sentiment patterns in over 3.1M social media posts using NLP techniques, TF-IDF vectorization, and a multi-class Logistic Regression model.

**Project Overview**

This project performs large-scale sentiment classification on 3,142,403 social media posts.

**It includes:**

```
Text cleaning & preprocessing
Tokenization
Stopword removal
Word frequency analysis
Sentiment mapping
TF-IDF vectorization
Machine learning classification (Positive, Negative, Neutral)
Performance evaluation
Visual exploration (bar charts & word clouds)
Analytical insights
```
This repository serves both as a portfolio project and a foundation for research in sentiment analysis, social media studies, communication science, and applied machine learning.

**Key Results**
```
✔ Model Accuracy: 79%
✔ Weighted F1-score: 0.79
✔ Learned patterns:
    Positive tweets → emotion, gratitude, humor, social bonding
    Negative tweets → stress, work pressure, routine frustration
    Neutral tweets → financial market updates & informational posts
```
Logistic Regression performed strongly on both positive and negative classes, with expected challenges on the underrepresented neutral class.

**Technologies Used**
```
Python
Pandas
NumPy
Scikit-Learn
Matplotlib
WordCloud
TF-IDF Vectorizer
Logistic Regression
```
**Pipeline Overview**
```
1. Data Cleaning
    removed URLs, punctuation, numbers
    standardized text
    removed stopwords
2. Sentiment Mapping
    0 → Negative
    1 → Positive
    2 → Neutral
3. Exploratory Analysis
    top words per sentiment
    sentiment distribution
    bar charts
    word clouds
4. Modeling
    TF-IDF vectorization with 100,000 features
    Train-test split (80/20)
    Logistic Regression with 2000 iterations
5. Evaluation
    Accuracy
    Precision
    Recall
    F1 score
    Classification report
```
**Machine Learning Model**
```
Algorithm: Logistic Regression
Feature Extraction: TF-IDF (100,000 features)
Train/Test Split: 80/20
Performance
Class          Precision   Recall	    F1-Score	 Support
Negative	    0.80	    0.77	      0.78	    314,658
Neutral	        0.76	    0.59	      0.67	     2,130
Positive	    0.77	    0.80	      0.79	    311,677
Weighted Avg	0.79	    0.79	      0.79	    628,465

Overall Accuracy: 0.79
```
This is strong performance for a 3-class NLP problem with real-world social media noise and extreme class imbalance (neutral class is very small).

**Insights (Summary)**
```
Positive tweets were dominated by emotional and expressive language
(“good”, “love”, “thanks”, “lol”), reflecting casual conversation, gratitude, and social bonding.

Negative tweets showed vocabulary related to frustration, obligation, and routine pressure
(“work”, “today”, “back”, “really”), suggesting dissatisfaction in everyday life.

Neutral tweets consisted primarily of financial and informational content
(“stocks”, “spx”, “aapl”, “https”), reflecting market updates and news-like posts.
```
The model successfully learned these patterns, achieving strong accuracy despite class imbalance.

**Repository Structure**
```
/notebook/
    └── sentiment_analysis.ipynb

/images/
    ├── wordcloud_positive.png
    ├── wordcloud_negative.png
    ├── wordcloud_neutral.png
    ├── barplot_positive.png
    ├── barplot_negative.png
    └── barplot_neutral.png

README.md
LICENSE
```
**Future Improvements**
```
Use deep learning (LSTM, BiLSTM, BERT)
Address class imbalance via resampling
Add bigram & trigram analysis
Build an interactive dashboard
```
**Author**

Prince Moulik
Communication Graduate • Aspiring NLP + Data Science Researcher
