# chatbot_tfidf.py
import json
import random
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Ensure required NLTK resources are available ----
# These downloads are safe to call repeatedly (they'll be no-ops if already present).
for pkg in ("punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        # ignore download errors here; program will likely work if resources already exist
        pass

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ---- Load intents (UTF-8) ----
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

intents = data.get("intents", [])
if not intents:
    raise ValueError("intents.json has no 'intents' key or is empty.")

# Build training lists: processed patterns, corresponding tags, and responses mapping
processed_patterns = []
pattern_tags = []
responses = {}  # tag -> list of responses

for intent in intents:
    tag = intent.get("tag")
    responses[tag] = intent.get("responses", [])
    for patt in intent.get("patterns", []):
        processed_patterns.append(preprocess_text(patt))
        pattern_tags.append(tag)

# If there are zero patterns (unlikely), include a tiny default to avoid errors
if len(processed_patterns) == 0:
    processed_patterns = ["hello"]
    pattern_tags = ["greeting"]
    responses.setdefault("greeting", ["Hello!"])

# ---- Vectorize ----
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_patterns)

# Threshold for confidence (tune between 0.2 and 0.35)
SIMILARITY_THRESHOLD = 0.30

def get_response(user_input: str) -> str:
    """
    Return a bot response for user_input. Also prints debug info to console.
    """
    if not user_input or not user_input.strip():
        return random.choice(responses.get("fallback", ["Please say something."]))

    processed = preprocess_text(user_input)
    user_vec = vectorizer.transform([processed])
    sims = cosine_similarity(user_vec, X)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    predicted_tag = pattern_tags[best_idx]

    # Debug print (appears in terminal where Flask is running)
    print(f"[DEBUG] Input: {user_input!r} | Processed: {processed!r} -> tag: {predicted_tag}, score: {best_score:.3f}")

    if best_score >= SIMILARITY_THRESHOLD:
        return random.choice(responses.get(predicted_tag, responses.get("fallback", ["Sorry."])))
    else:
        return random.choice(responses.get("fallback", ["I'm not sure I understood — can you rephrase?"]))


if __name__ == "__main__":
    # quick CLI test
    print("Chatbot CLI — type 'quit' to exit")
    while True:
        u = input("You: ").strip()
        if u.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        print("Bot:", get_response(u))
