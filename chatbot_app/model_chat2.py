import numpy as np
import random 
import json
import pickle
import keras
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping  # Import EarlyStopping

# Télécharger les ressources nécessaires
# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Charger les intents
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Préparer les données
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatization et nettoyage
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Sauvegarder les données préparées
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Préparer l'entraînement
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
    
    # Créer le "bag of words"
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)

    # Créer l'output pour la classification
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

# Split data into training and testing sets
train_size = int(len(training) * 0.8)  # 80% for training
test_size = len(training) - train_size
train_data = training[:train_size]
test_data = training[train_size:]

train_X = train_data[:, :len(words)]
train_Y = train_data[:, len(words):]
test_X = test_data[:, :len(words)]
test_Y = test_data[:, len(words):]

# Définir le modèle Keras
model = keras.Sequential()

model.add(keras.layers.Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(train_Y[0]), activation='softmax'))

# Optimiseur
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)

# Entraînement du modèle
history = model.fit(
    np.array(train_X), np.array(train_Y),
    epochs=500, batch_size=100, verbose=1,
    validation_data=(test_X, test_Y),
    callbacks=[early_stopping]  # Add early stopping here
)
# Sauvegarder le modèle
model.save('chatbot_maryIPK.keras')

print("Model training completed and saved!")

# Plot learning curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model_accuracy.png')  # Save the accuracy plot
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_loss.png')  # Save the loss plot
plt.show()