import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- STEP 1: LOAD DATA ---
print("Loading data...")
# We only need 'text' (the tweet) and 'target' (1=Real Disaster, 0=Fake)
df = pd.read_csv('train.csv')

# Quick clean: drop rows with missing text (if any)
df = df.dropna(subset=['text'])

print(f"Total tweets: {len(df)}")
print("Example Tweet:", df['text'].iloc[0])
print("Target:", df['target'].iloc[0])

# --- STEP 2: PREPROCESSING (The NLP Part) ---
print("\nTokenizing text...")

# Hyperparameters
vocab_size = 10000    # We will only look at the top 10,000 words
max_length = 100      # Crop tweets to 100 words
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"     # For words we haven't seen before

# Tokenizer: Converts words to numbers
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['text'])

# Convert text to sequences of numbers
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Split into Train/Test (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(padded, df['target'], test_size=0.2, random_state=42)

# --- STEP 3: BUILD THE DEEP LEARNING MODEL ---
print("\nBuilding LSTM Neural Network...")

model = Sequential([
    # 1. Embedding Layer: Converts number sequences into dense vectors (understanding meaning)
    Embedding(vocab_size, 16, input_length=max_length),
    
    # 2. Bidirectional LSTM: Reads text forwards AND backwards (crucial for context)
    Bidirectional(LSTM(32)),
    
    # 3. Dense Layers: The decision making
    Dense(24, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(1, activation='sigmoid') # Output: 0 or 1
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- STEP 4: TRAIN ---
print("Training Model (This uses Deep Learning)...")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

# --- STEP 5: EVALUATE ---
print("\nEvaluating...")
y_pred_prob = model.predict(X_test)
# Convert probabilities to 0 or 1
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\nDone! This proves you can use TensorFlow and NLP.")