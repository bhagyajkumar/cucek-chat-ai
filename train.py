import json
import numpy as np
import random
import pickle
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

# Define intents
with open("data.json", "r") as f:
    intents = json.loads(f.read())
# NLP Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(sentence):
    """Tokenizes, stems, and removes stopwords from a sentence."""
    words = word_tokenize(sentence.lower())
    words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return " ".join(words)

# Prepare dataset
corpus = []
labels = []
all_patterns = []
all_labels = []

for intent, data in intents.items():
    for pattern in data["patterns"]:
        processed_pattern = preprocess(pattern)
        all_patterns.append(processed_pattern)
        all_labels.append(intent)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(all_patterns)
word_index = tokenizer.word_index

# Convert texts to sequences
X = tokenizer.texts_to_sequences(all_patterns)
X = pad_sequences(X, padding="post")

# Convert labels to numbers
label_to_index = {label: idx for idx, label in enumerate(set(all_labels))}
y = np.array([label_to_index[label] for label in all_labels])

# Build LSTM Model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=X.shape[1]),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(len(label_to_index), activation="softmax")
])

# Compile Model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
model.fit(X, y, epochs=200, verbose=1)

# Save Model & Tokenizer
model.save("chatbot_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_to_index.pkl", "wb") as f:
    pickle.dump(label_to_index, f)

# Context Tracking
context = None

# Chatbot Response Function
def chatbot_response(user_input):
    global context

    # Load saved model & tokenizer
    model = tf.keras.models.load_model("chatbot_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_to_index.pkl", "rb") as f:
        label_to_index = pickle.load(f)

    # Preprocess input
    processed_input = preprocess(user_input)
    input_seq = tokenizer.texts_to_sequences([processed_input])
    input_seq = pad_sequences(input_seq, maxlen=X.shape[1], padding="post")

    # Predict intent
    prediction = model.predict(input_seq)
    predicted_label = list(label_to_index.keys())[np.argmax(prediction)]

    # Handle confirmation
    confirmations = ["yes", "yeah", "sure", "ok", "okay"]
    if processed_input in confirmations and context:
        for sub_intent, data in intents.items():
            if "parent" in data and data["parent"] == context:
                context = None
                return random.choice(data["responses"])

    # Handle sub-intents
    if context:
        for sub_intent, data in intents.items():
            if "parent" in data and data["parent"] == context:
                if processed_input in [preprocess(p) for p in data["patterns"]]:
                    context = None
                    return random.choice(data["responses"])

    # Update context if needed
    if "context" in intents[predicted_label]:
        context = intents[predicted_label]["context"]

    return random.choice(intents[predicted_label]["responses"])

# Run chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
