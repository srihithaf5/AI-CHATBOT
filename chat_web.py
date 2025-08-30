from flask import Flask, render_template, request, jsonify
from chatbot_tfidf import get_response  # import your function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # simple page with JS that posts user input

@app.route('/ask', methods=['POST'])
def ask():
    q = request.json.get('message', '')
    return jsonify({'reply': get_response(q)})

if __name__ == '__main__':
    app.run(debug=True)
