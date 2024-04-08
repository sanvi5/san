# Text Data Preprocessing Lib
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

# Load the stemmer
ps = PorterStemmer()

# words to be ignored/omitted while framing the dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

import json
import pickle
import numpy as np

# Model Load Lib
import tensorflow
from tensorflow.keras.models import load_model

# Load the model
model = load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

def preprocess_user_input(user_input):
    bag = [0] * len(words)  # Initialize the bag of words with zeros

    # Tokenize the user input
    user_tokens = word_tokenize(user_input)

    # Convert the user input into its root words (stemming) and remove ignored words
    stemmed_words = [ps.stem(word.lower()) for word in user_tokens if word not in ignore_words]

    # Loop through words and set 1 for each word in the bag
    for idx, w in enumerate(words):
        if w in stemmed_words:
            bag[idx] = 1

    return np.array([bag])  # Return the bag of words as a numpy array

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction)
    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    
    # Extract the class from the predicted_class_label
    predicted_class = classes[predicted_class_label]

    # Find the appropriate intent
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            # Choose a random bot response
            bot_response = random.choice(intent['responses'])
            return bot_response

print("Hi, I am Stella. How can I help you?")

while True:
    # Take input from the user
    user_input = input('You: ')

    # Get the bot's response
    response = bot_response(user_input)
    print("Stella: ", response)
