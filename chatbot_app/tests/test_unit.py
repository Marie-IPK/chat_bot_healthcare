import pytest
import sys
import os
from chatbot_flask import app, predict_class, get_response, bag_of_words, clean_up_sentence
import unittest
import json
import numpy as np
import pickle
import keras
# Ajouter le chemin de `chatbot_app` au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire contenant unit_test.py
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Répertoire parent (chatbot_app)
sys.path.append(parent_dir)



@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_home_status_code(client):
    response = client.get('/')
    assert response.status_code == 200

class TestChatbotModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Chemins des fichiers nécessaires
        cls.model_path = os.path.join(os.getcwd(), 'chatbot_maryIPK.keras')
        cls.words_path = os.path.join(os.getcwd(), 'words.pkl')
        cls.classes_path = os.path.join(os.getcwd(), 'classes.pkl')
        cls.intents_path = os.path.join(os.getcwd(), 'intents.json')

        # Vérifier que les fichiers nécessaires existent
        for path in [cls.model_path, cls.words_path, cls.classes_path, cls.intents_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier introuvable : {path}")

        # Charger les fichiers nécessaires
        cls.model = keras.models.load_model(cls.model_path)
        cls.words = pickle.load(open(cls.words_path, 'rb'))
        cls.classes = pickle.load(open(cls.classes_path, 'rb'))
        with open(cls.intents_path, 'r') as file:
            cls.intents_data = json.load(file)

    def test_model_loading(self):
        """Test que le modèle Keras est correctement chargé."""
        self.assertIsNotNone(self.model, "Le modèle Keras n'a pas été chargé correctement.")

    def test_bag_of_words(self):
        """Test que la fonction bag_of_words produit les bons résultats."""
        sentence = "Hello, how are you?"
        bow = bag_of_words(sentence)
        self.assertEqual(len(bow), len(self.words), "La taille du bag of words ne correspond pas au nombre de mots.")

    def test_predict_class(self):
        """Test que la fonction predict_class retourne une prédiction valide."""
        sentence = "What can you do?"
        intents = predict_class(sentence)
        self.assertGreater(len(intents), 0, "La prédiction doit retourner au moins un intent.")
        self.assertIn('intent', intents[0], "La prédiction doit contenir un champ 'intent'.")
        self.assertIn('probability', intents[0], "La prédiction doit contenir un champ 'probability'.")

    def test_get_response(self):
        """Test que la fonction get_response retourne une réponse valide."""
        intents = [{'intent': 'greeting', 'probability': '0.95'}]
        response = get_response(intents, self.intents_data)
        self.assertIsInstance(response, str, "La réponse doit être une chaîne de caractères.")
        self.assertGreater(len(response), 0, "La réponse ne doit pas être vide.")

    def test_clean_up_sentence(self):
        """Test que la fonction clean_up_sentence nettoie correctement une phrase."""
        sentence = "Hello, how are you?"
        cleaned = clean_up_sentence(sentence)
        self.assertIsInstance(cleaned, list, "Le résultat doit être une liste.")
        self.assertTrue(all(isinstance(word, str) for word in cleaned), "Chaque mot nettoyé doit être une chaîne.")

    def test_flask_app_home(self):
        """Test que la route '/' de l'application Flask fonctionne correctement."""
        with app.test_client() as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200, "La route '/' devrait retourner un statut 200.")

    def test_flask_app_get_response(self):
        """Test que la route '/get' retourne une réponse valide."""
        with app.test_client() as client:
            response = client.get('/get', query_string={'msg': 'Hi'})
            self.assertEqual(response.status_code, 200, "La route '/get' devrait retourner un statut 200.")
            data = response.get_json()
            self.assertIn('response', data, "La réponse JSON doit contenir une clé 'response'.")
            self.assertIsInstance(data['response'], str, "La réponse doit être une chaîne de caractères.")

if __name__ == '__main__':
    unittest.main()


