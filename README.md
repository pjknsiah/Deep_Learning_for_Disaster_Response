# NLP-Based Disaster Response Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Context](https://img.shields.io/badge/AI-Social%20Good-purple)

A Deep Learning Natural Language Processing (NLP) pipeline designed to filter critical information from noise during humanitarian crises. Using **Bidirectional LSTMs** and **Word Embeddings**, this system classifies social media posts as "Real Disasters" or "Metaphorical Noise" with **79% accuracy**.

## 🚀 Project Overview

During emergencies (floods, fires, earthquakes), social media is a vital source of real-time situational awareness. However, keyword-based filtering fails because words like "ablaze," "panic," or "avalanche" are often used metaphorically (e.g., *"The sunset was ablaze with color"*).

This project solves the **Semantics vs. Keyword** problem by using Deep Learning to understand context.
* **Goal:** Automatically flag tweets that require immediate humanitarian attention.
* **Tech Stack:** Python, TensorFlow (Keras), Scikit-Learn, Pandas.
* **Architecture:** Embedding Layer $\to$ Bidirectional LSTM $\to$ Dense Output.

## 📊 Key Results

* **Dataset:** Kaggle NLP with Disaster Tweets (10,000+ samples).
* **Model Architecture:** Bidirectional LSTM (Long Short-Term Memory).
* **Accuracy:** **79.20%** on the validation set.
* **Why LSTM?** Unlike standard Bag-of-Words models, LSTMs preserve the *sequence* of words, allowing the model to understand that *"The movie was a disaster"* is negative sentiment, not an emergency event.

### Engineering Decisions
* **Embeddings:** Implemented a learned embedding layer (16-dim) to capture semantic relationships between words.
* **Sequence Handling:** Utilized `pad_sequences` to standardize tweet lengths to 100 tokens, ensuring uniform tensor input.
* **Regularization:** Applied `Dropout(0.5)` to prevent overfitting on the relatively small dataset.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* pip

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/disaster-response-nlp.git](https://github.com/YOUR_USERNAME/disaster-response-nlp.git)
cd disaster-response-nlp