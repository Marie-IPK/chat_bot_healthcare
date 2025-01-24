import pytest
import sys
import os
import time  # Pour mesurer le temps

# Ajouter le chemin de `chatbot_app` au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire contenant le fichier de test
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Répertoire parent (chatbot_app)
sys.path.append(parent_dir)

from chatbot_flask import app  # Importer le fichier Flask

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_response_time(client):
    """Test du temps de réponse de l'endpoint `/`"""
    start_time = time.time()  # Début de la mesure du temps
    response = client.get('/')
    end_time = time.time()  # Fin de la mesure du temps

    # Calcul du temps écoulé
    elapsed_time = end_time - start_time

    # Assurez-vous que le temps de réponse est inférieur à une limite (par ex., 0.5 seconde)
    assert elapsed_time < 0.5, f"Temps de réponse trop long : {elapsed_time} secondes"

    # Vérifiez également que le statut est correct
    assert response.status_code == 200
