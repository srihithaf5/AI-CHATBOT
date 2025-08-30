import json
import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------
# Preprocess text
# -------------------------
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in string.punctuation and word not in stop_words
    ]
    return " ".join(tokens)


# -------------------------
# Load intents.json
# -------------------------
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)["intents"]

patterns = []
responses = []
tags = []

for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        responses.append(intent["responses"])
        tags.append(intent["tag"])

# Train TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

print("Chatbot ready! Type 'quit' or 'exit' to stop.")

# -------------------------
# Chat loop
# -------------------------
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye! Have a great day 😊")
        break

    processed_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vector, X)
    max_similarity = similarities.max()
    index = similarities.argmax()

    # Debugging info
    print(f"[DEBUG] Predicted intent: {tags[index]}, similarity: {max_similarity:.2f}")

    if max_similarity > 0.3:  # lowered threshold
        bot_response = random.choice(responses[index])
        print("Bot:", bot_response)
    else:
        print("Bot: I'm not sure I understood — can you rephrase?")
