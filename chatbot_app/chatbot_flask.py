import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras
from flask import Flask, request, jsonify, render_template

# Load intents.json data at the start
with open('intents.json') as file:
    intents_data = json.load(file)

# Load model and other resources
lemmatizer = WordNetLemmatizer()
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbot_maryIPK.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in result]

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_input = request.args.get('msg')
    intents = predict_class(user_input)

    if intents:
        res = get_response(intents, intents_data)
    else:
        res = "Sorry, I didn't understand that."
    
    return jsonify({'response': res})

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't understand that."

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8080)
