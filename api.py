from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import nltk



class ChatInputModel(BaseModel):
    input:str

class ChatOutputModel(BaseModel):
    input:str
    output:str

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

def chatbot_response(user_input):

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

    return random.choice(intents[predicted_label]["responses"])

@app.post("/chat", response_model=ChatOutputModel)
def chat_route(from_user:ChatInputModel):
    output = chatbot_response(from_user.input)
    chat_response = ChatOutputModel(
        input=from_user.input,
        output=output
    )
    return chat_response
