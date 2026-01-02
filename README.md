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
git clone https://github.com/pjknsiah/Deep_Learning_for_Disaster_Response.git
cd Deep_Learning_for_Disaster_Response
```

### 2. Install Dependencies
```bash
pip install pandas numpy tensorflow scikit-learn
```

## 🏃 Usage

Run the training pipeline:
```bash
python disaster_nlp.py
```

The script will:
1. Load the `train.csv` dataset.
2. Tokenize and pad the tweets.
3. Train the Bidirectional LSTM model.
4. Output the model accuracy and classification report.

## 📁 Project Structure

* `disaster_nlp.py`: The main Python script.
* `train.csv`: The training dataset.
* `README.md`: Project documentation.

## 🧠 Model Architecture

The model is built using **TensorFlow/Keras** and consists of the following layers:

1.  **Embedding Layer**:
    *   Input: Integer-encoded vocabulary (Top 10,000 words).
    *   Output: Dense vector of size 16 for each word.
    *   *Purpose:* Captures semantic similarity between words (e.g., "fire" and "blaze" have similar vectors).

2.  **Bidirectional LSTM (32 units)**:
    *   Input: Sequence of word vectors.
    *   *Purpose:* Processes the text in both directions (start-to-end and end-to-start) to understand the full context of a phrase.

3.  **Dense Layer (24 units, ReLU)**:
    *   *Purpose:* Fully connected layer to interpret the features extracted by the LSTM.

4.  **Dropout (0.5)**:
    *   *Purpose:* Randomly sets 50% of input units to 0 during training to prevent overfitting and improve generalization.

5.  **Output Layer (1 unit, Sigmoid)**:
    *   *Purpose:* Outputs a probability score between 0 and 1 (0 = Not Disaster, 1 = Real Disaster).

## 🔮 Future Improvements

POTENTIAL AREAS FOR DEVELOPMENT

*   **Pre-trained Embeddings**: Replace the learned embedding layer with **GloVe** or **Word2Vec** to leverage transfer learning from massive corpora.
*   **Transformer Models**: Implement **BERT** or **RoBERTa** for state-of-the-art NLP performance.
*   **Hyperparameter Tuning**: Use grid search to optimize the learning rate, batch size, and number of LSTM units.
*   **Data Augmentation**: Increase the dataset size using synonym replacement or back-translation techniques.