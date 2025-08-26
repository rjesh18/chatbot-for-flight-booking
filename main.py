import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
#nltk.download('punkt')
#nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load your intents JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', "'"]

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the utterance
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent['tag']))
        # Add to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Number of documents: {len(documents)}")
print(f"Number of classes: {len(classes)}")
print(f"Number of unique words: {len(words)}")

# Create training data - FIXED APPROACH
train_x = []
train_y = []

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0] if word not in ignore_words]
    
    # Create bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Create output row
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    
    train_x.append(bag)
    train_y.append(output_row)

# Convert to numpy arrays
train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"Sample train_x: {train_x[0] if len(train_x) > 0 else 'Empty'}")
print(f"Sample train_y: {train_y[0] if len(train_y) > 0 else 'Empty'}")

# Build the model with correct input shape
model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
try:
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    
    # Save the model and data
    model.save('chatbot_model.h5')
    pickle.dump({
        'words': words, 
        'classes': classes, 
        'ignore_words': ignore_words,
        'lemmatizer': lemmatizer
    }, open('training_data.pkl', 'wb'))
    
    print("Model training completed successfully!")
    
except Exception as e:
    print(f"Error during training: {e}")
    print("Debug info:")
    print(f"train_x shape: {train_x.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"Number of words: {len(words)}")
    print(f"Number of classes: {len(classes)}")