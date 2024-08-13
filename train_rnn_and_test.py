import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional, Dense, LSTM)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

train_data = pd.read_csv('./datasets/traindata.csv')
test_data = pd.read_csv('./datasets/testdata.csv')

def pretokenize_pd(inp):
    rgx_html_entities = r'&[^;\s]*;'
    rgx_unicode = r'[^\x00-\x7F]+'
    rgx_nums = r'\d+'
    rgx_punct = r'[^\w\s]'

    rgx_all = f"({rgx_html_entities})|({rgx_unicode})|({rgx_nums})|({rgx_punct})"

    cleaned = []

    for tweet in inp:
        tweet = tweet.lower()
        cleaned_rgx = re.sub(rgx_all, '', tweet)
        cleaned.append(cleaned_rgx)
    
    return cleaned

train_data['cleaned_tweet'] = pretokenize_pd(train_data['tweet'])
test_data['cleaned_tweet'] = pretokenize_pd(test_data['tweet'])        

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['cleaned_tweet'])

print("TOKENIZER INDEX: \n", tokenizer.word_index)
train_sequences = tokenizer.texts_to_sequences(train_data['cleaned_tweet'])
test_sequences = tokenizer.texts_to_sequences(test_data['cleaned_tweet'])

train_padded = pad_sequences(train_sequences, maxlen=max_len)
test_padded = pad_sequences(test_sequences, maxlen=max_len)

labels = np.array(train_data['label'])

model = Sequential()
model.add(Embedding(max_words, 16, input_shape=(max_len,)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(train_padded, labels, epochs=7, batch_size=32)

def test_random(samples=200):
    sample_indices = np.random.choice(len(test_data), size=samples, replace=False)
    test_samples = test_data.iloc[sample_indices]['cleaned_tweet']

    test_sequences = tokenizer.texts_to_sequences(test_samples)
    test_padded = pad_sequences(test_sequences, maxlen=max_len)

    predictions = model.predict(test_padded)

    out_tweets_0 = []
    out_predicts_0 = []

    out_tweets_1 = []
    out_predicts_1 = []

    for i, (tweet, prediction) in enumerate(zip(test_samples, predictions)):
        if prediction[0] <= 0.50:
            out_tweets_0.append(tweet)
            out_predicts_0.append(prediction[0])
        else:
            out_tweets_1.append(tweet)
            out_predicts_1.append(prediction[0])

    print("Non-hateful (less than or eq 0.50):")
    b=0

    print("Non-hateful (less than or eq 0.50):")
    for tweet, prediction in zip(out_tweets_0, out_predicts_0):
        print(f"'{tweet}' -> {'{:.6f}'.format(prediction)}")

    print("\n_______________________\n")

    print("Hateful (over 0.50):")
    for tweet, prediction in zip(out_tweets_1, out_predicts_1):
        print(f"'{tweet}' -> {'{:.6f}'.format(prediction)}")

test_random()

model.save('model_new.keras')

try:
    model.save('model_new.h5')
except ValueError as e:
    print("h5 model already exists.")