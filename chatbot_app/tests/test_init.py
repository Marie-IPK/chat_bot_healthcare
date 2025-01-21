import pytest

from chatbot_flask import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_home_status_code(client):
    response = client.get('/')
    assert response.status_code == 200
