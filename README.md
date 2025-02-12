# chat_bot_healthcare

# Chatbot Project

This repository contains the implementation of a chatbot using Python and deep learning techniques. The chatbot is designed to interact with users and provide relevant responses based on trained intents.
## Virtual environmenet 
- **Install virtual environmenet :**
  On Linux
  `pip install virtualenv`
- **Create virtual environment :**
  On Linux
  `virtualenv "name_virtual_env"`
- **Activate virtual environment :**
  On Linux
  `source name_virtual_env/bin/activate`
## Features

- **Natural Language Processing (NLP):** Processes user input using lemmatization and tokenization.
- **Deep Learning:** Uses a neural network to classify user intents and generate responses.
- **Customizable:** Easily extendable with new intents and responses.
- **Bag of Words (BoW):** Implements a bag-of-words model for text preprocessing.

## Prerequisites

Before running the chatbot, ensure the following are installed:

- Python 3
- `nltk`
- `numpy`
- `keras`
- `tensorflow`

You can install the dependencies using:

```bash
pip install -r requirements.txt
```
## Project structure 

```bash
CCHAT_BOT_HEALTHCARE
├── .github/
│ └── workflows/
│ └── python-app.yml
├── chatbot_app/
│ ├── pycache/
│ │ └── chat_2.cpython-312.pyc
│ ├── nltk_data/
│ │ └── tokenizers/
│ │ └── punkt/
│ ├── static/
│ │ └── logo.png
│ ├── templates/
│ │ └── index.html
│ ├── chat_2.py
│ ├── chatbot_flask.py
│ ├── chatbot_marylPK.keras
│ ├── classes.pkl
│ ├── Dockerfile
│ ├── intents.json
│ ├── model_accuracy.png
│ ├── model_chat2.py
│ ├── model_loss.png
│ ├── requirements.txt
│ ├── words.pkl
│ └── LICENSE
├── chatbot_env/
│ ├── bin/
│ ├── lib/
│ ├── share/
│ └── pyvenv.cfg
├── .gitignore
└── README.md                 # Project documentation                 

```
## Usage 
### training the model 
To train the chatbot model, ensure your intents.json is configured with your data, and run:
```bash
python3 -u model_chat2.py
```
### Running the chatbot
Start the chatbot using:
```bash 
python3 -u chat_2.py
```
### Adding intents
To add new intents:

Update the intents.json file with your new intent and responses.
Retrain the model using train.py.
Example entry in intents.json:
```bash
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi", "Good morning"],
      "responses": ["Hi there!", "Hello!", "How can I help you?"]
    }
  ]
}
```
## Troubleshooting
### Common Errors
- ValueError: Input shape mismatch. Ensure your bag-of-words vector matches the model's input size.
- ModuleNotFoundError: Install missing dependencies using pip install -r requirements.txt.

### Contributing
We welcome contributions! If you’d like to improve the project, follow these steps:

- Fork this repository.
- Create a new branch (git checkout -b feature-branch-name).
- Commit your changes (git commit -m "Add new feature").
- Push to the branch (git push origin feature-branch-name).
- Open a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for more information.
