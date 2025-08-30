from flask import Flask, render_template, request, jsonify
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "Sorry, I didn’t understand that."})

    # Simple matching
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_message.lower():
                return jsonify({"response": random.choice(intent["responses"])})

    return jsonify({"response": "I'm not sure about that. Can you rephrase?"})

if __name__ == "__main__":
    app.run(debug=True)
