import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import streamlit as st
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents and model data
intents = json.loads(open(r'C:\Users\Rajesh\chatbot\intents.json').read())

# Load the training data - make sure you're loading the correct pickle file
# If you saved it as 'training_data.pkl' during training, load that instead
data = pickle.load(open('training_data.pkl', 'rb'))
words = data['words']
classes = data['classes']
ignore_words = data['ignore_words']

# Load the model
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    # Convert to lowercase and filter ignore words - THIS IS CRITICAL
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # Debug: print what words were found
    if show_details:
        print(f"Input words: {sentence_words}")
        print(f"Vocabulary size: {len(words)}")
    
    # Create bag of words with correct length
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Matched word: {w}")
    
    if show_details:
        print(f"Bag of words (non-zero indices): {[i for i, val in enumerate(bag) if val == 1]}")
    
    return np.array(bag, dtype=np.float32)

def predict_class(sentence, model):
    # Generate the bag of words
    p = bow(sentence, words, show_details=True)  # Set to True for debugging
    
    print(f"Input shape to model: {p.shape}")
    print(f"Model expected input shape: {model.input_shape}")
    
    # Reshape to match model input shape (1, vocabulary_size)
    p = p.reshape(1, -1)
    
    # Predict
    res = model.predict(p, verbose=0)[0]
    
    # Filter predictions
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that. Could you try rephrasing?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:  # Changed from 'tag' to 'name' to match your JSON structure
            result = random.choice(i['responses'])
            return result
    else:
        result = "I don't understand that yet. Could you try asking differently?"
    
    return result

print("GO! Bot is running!")
print(f"Vocabulary size: {len(words)}")
print(f"Classes: {classes}")


# ----- Streamlit App -----

st.set_page_config(page_title="Flight Booking Chatbot", page_icon="✈️", layout="centered")

st.title("✈️ Flight Booking Chatbot")
st.write("Hi, I'm your AI assistant! Ask me about flight bookings, schedules, and more.")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Get response
    intents_pred = predict_class(user_input, model)
    bot_response = get_response(intents_pred, intents)

    # Save conversation
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "You":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)
