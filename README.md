# AI-CHATBOT

Project Overview

This project is a simple AI-powered chatbot built using Python, Flask, and Natural Language Processing (NLP) techniques. The chatbot is designed to understand user inputs, process them, and provide appropriate responses based on predefined intents stored in a JSON file (intents.json). It demonstrates the basic workflow of a conversational AI system, where text classification and intent recognition are used to provide meaningful responses.

The chatbot runs on a Flask web server and provides a web-based user interface through index.html. Users can type queries into the browser, and the chatbot will respond in real-time. It is a beginner-friendly project that showcases how natural language processing can be applied in practical chatbot applications.

FEATURES

1'Interactive Chat Interface
The chatbot comes with a clean and simple web interface built using HTML, CSS, and JavaScript. Users can type messages and instantly see responses from the bot.

2.Intent Recognition
User queries are matched with predefined intents using TF-IDF vectorization and cosine similarity, allowing the chatbot to understand user inputs even when they are phrased differently.

3.Customizable Knowledge Base
All chatbot responses are stored in intents.json, which can be easily modified or expanded to include new categories, patterns, and responses.

4.Lightweight & Easy to Run
The project only requires Python and a few dependencies like Flask, scikit-learn, and NLTK, making it easy to set up and run on any system.

5.Extensible Design
Developers can extend the chatbot by integrating APIs (e.g., weather, news, or stock prices), adding voice input/output, or connecting to a database.

INSTALLATION & SETUP

Clone the repository or download the source code.

Create and activate a virtual environment:

python -m venv venv

venv\Scripts\activate    # On Windows

Install dependencies:

pip install -r requirements.txt

Run the Flask app:

python app.py

Open your browser and go to:

http://127.0.0.1:5000

How It Works

User enters a message in the chatbox.

The message is sent to the Flask backend via a POST request.

The backend preprocesses the text using NLTK stopwords and tokenization.

The text is converted into vectors using TF-IDF.

The system compares the input with patterns in intents.json using cosine similarity.

The best-matching intent is selected, and a response is returned.

The chatbot displays the response in the web interface.

Future Improvements

Add machine learning models (e.g., neural networks) for improved accuracy.

Integrate with speech-to-text and text-to-speech for voice interaction.

Add support for external APIs like weather updates, Wikipedia search, or calculators.

Store conversations in a database for analysis and learning.

Deploy the chatbot on cloud platforms like Heroku, AWS, or Render.

Conclusion

This chatbot project is a hands-on introduction to NLP and web-based AI applications. It is suitable for students, beginners in AI, or developers who want to explore conversational interfaces. By modifying the intents.json file, you can easily customize the chatbot for different domains like customer support, education, healthcare, or entertainment.

The simplicity of the project makes it a solid foundation for learning, while its modular design allows easy upgrades into more advanced conversational AI systems. Whether you want to create a personal assistant, a FAQ bot, or just explore AI, this project serves as a stepping stone into the world of intelligent chatbots.

OUTPUTS: 













