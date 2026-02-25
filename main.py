"""Sentence completion: load model and generate text from your seed. Run from terminal: python main.py"""

import pickle
import string
import tensorflow as tf
import numpy as np

#----------------------------------------------------------- Load the saved model
lstm_model = tf.keras.models.load_model('lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('config.pkl', 'rb') as f:
    config = pickle.load(f)

max_length = config['max_length']
index_to_word = config['index_to_word']
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# ------------------------------------------------------- PREPROCESS
no_punct = str.maketrans('', '', string.punctuation)
def preprocess(text):
    return text.lower().translate(no_punct)

# ------------------------------------------------------- PREDICTOR
def predictor(model, tokenizer, text, max_len):
    text = preprocess(text)
    seq = tokenizer.texts_to_sequences([text])[0]
    seq_padded = pad_sequences([seq], maxlen=max_len, padding='pre')
    
    # Model gives probabilities for each word; we take the one with highest probability
    pred = model.predict(seq_padded, verbose=0)
    pred_index = np.argmax(pred[0])  
    word_index_to_use = pred_index + 1
    if word_index_to_use in index_to_word:
        return index_to_word[word_index_to_use]
    return ''

# ------------------------------------------------------- GENEERATE TEXT
def generate_text(model, tokenizer, seed_text, max_len, num_words):
    for i in range(num_words):
        next_word = predictor(model, tokenizer, seed_text, max_len)
        if next_word == '':
            break
        seed_text = seed_text + ' ' + next_word
    return seed_text

# ------------------------------------------------------- MAIN FUNCTION
if __name__ == "__main__":
    seed_text = input("Enter a seed text: ").strip()
    while not seed_text:
        print("Enter at least one word (e.g. life is)")
        seed_text = input("Enter a seed text: ").strip()
    generated = generate_text(lstm_model, tokenizer, seed_text, max_length, 10)
    print("Generated:", generated)