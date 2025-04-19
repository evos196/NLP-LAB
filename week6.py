import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

corpus = ["The cat sat on the mat"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

vocabulary_size = total_words

skip_grams = [skipgrams(sequence, vocabulary_size, window_size=5) for sequence in tokenizer.texts_to_sequences(corpus)]
pairs, labels = skip_grams[0][0], skip_grams[0][1]

embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=1))

model.add(Reshape((embedding_dim,)))
model.add(Dense(units=total_words, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, num_classes=total_words)
model.fit(np.array(pairs)[:,0], labels, epochs=10, batch_size=32)

word_embeddings = model.get_layer(index=0).get_weights()[0]

for word, token in tokenizer.word_index.items():
  print(f"{word}:{word_embeddings[token]}")
  
