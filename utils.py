# Utilities
from keras.preprocessing import text, sequence
import numpy as np


def train_test_split(data, test_size, random_state):
  '''splits data into test and train set'''
  
  np.random.seed(random_state)
  
  n_items = data.shape[0]

  size = int(test_size*n_items)
  inds = np.random.permutation(np.arange(n_items))

  test_data = data.iloc[inds[:size]]
  train_data = data.iloc[inds[size:]]
  
  return train_data, test_data



def to_sequences(x_train, x_test, maxlen):
  ''' tokenizes text data and applies padding to it
  x_train and x_test: pandas DataFrame or numpy array
  maxlen: int, maximum length of sequences
  '''
  tokenizer = text.Tokenizer()
  tokenizer.fit_on_texts(x_train)

  sequences_train = tokenizer.texts_to_sequences(x_train)
  sequences_test = tokenizer.texts_to_sequences(x_test)
  
  # number of different words in the dataset
  vocab_size = len(tokenizer.word_index) + 1

  sequences_train = sequence.pad_sequences(sequences_train,padding='post', maxlen=maxlen)
  sequences_test = sequence.pad_sequences(sequences_test,padding='post', maxlen=maxlen)

  return sequences_train, sequences_test, vocab_size
