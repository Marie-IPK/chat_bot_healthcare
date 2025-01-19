
import random 
import json 
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = keras.models.load_model('chatbot_maryIPK.keras')

# Fonction de nettoyage de la phrase
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Fonction pour créer un bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w: 
                bag[i] = 1
    return np.array(bag)

# Fonction pour prédire la classe
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1  # Baisse du seuil pour améliorer la reconnaissance
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)

    # Debugging : Afficher les résultats de la prédiction
    print(f"Prediction results for '{sentence}': {result}")
    
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in result]

# Fonction pour obtenir la réponse du chatbot
def get_response(intents_list, intents_json):
    if not intents_list:  # Si la liste est vide
        return print("Sorry, I didn't understand that.")
    
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

print("Great! Bot is Running")

# Boucle du chatbot
while True: 
    message = input("")
    ints = predict_class(message)
    
    # Vérifier si ints est vide
    if not ints:
        print("Sorry, I didn't understand that.")
        continue
    
    res = get_response(ints, intents)
    print(res)
