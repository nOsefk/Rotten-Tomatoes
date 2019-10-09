
from __future__ import print_function #Python 2/3 compatibility for print statements
import pandas as pd
pd.set_option('display.max_colwidth', 170)
import spacy
from spacy.tokenizer import Tokenizer



df = pd.read_csv('train.tsv', encoding= 'UTF-8', sep='\t')
df_train = df[['Phrase']]
df_train.head()

nlp = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)


def text_to_token(text_seqs):
    token_seqs = [[word.lower_ for word in tokenizer(text_seq)] for text_seq in text_seqs]
    return token_seqs


df_train['Tokenized_Phrase'] = text_to_token(df_train['Phrase'])


import pickle 
from collections import Counter


def token_count(token_seqs):
    word_freqs={}  
    for token_seq in token_seqs :  
        word_freq = Counter(token_seq)
        for key in word_freq:
            if key in word_freqs :
                word_freqs[key] += word_freq[key]
            else :
                word_freqs[key] = word_freq[key]
    return word_freqs


def make_lexicon(token_seqs, min_freq=1):
    count = token_count(token_seqs)
    lexicon = {k:v for k,v in count.items() if v >= min_freq}
    lexicon = {token:idx + 2 for idx,token in enumerate(lexicon)}
    lexicon['<UNK>'] = 1 # Unknown words are those that occur fewer than min_freq times
    print("LEXICON SAMPLE ({} total items).".format(len(lexicon)))
    return lexicon
    ###END 

lexicon = make_lexicon(df_train['Tokenized_Phrase'])

def tokens_to_vecor(token_seqs):
    token_seqs = [[token.vector for token in tokenizer(token_seq)] for token_seq in token_seqs]
    return token_seqs


df_train['Phrase_Vectors'] = tokens_to_vecor(df_train['Phrase'])


df_train[['Tokenized_Phrase', 'Phrase_Vectors']][:10]


max_seq_len = df_train['Phrase_Vectors'].map(lambda x: len(x)).max()

from keras.preprocessing.sequence import pad_sequences

def pad_vectors(vec_seqs, max_seq_len):
    padded_vec = pad_sequences(vec_seqs, maxlen=max_seq_len + 1)
    return padded_vec


train_padded = pad_vectors(df_train['Phrase_Vectors'], max_seq_len)

train_padded[0].shape

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers import Input, Dense, TimeDistributed, Flatten
from keras.models import Model


def create_rnn_model(n_input_nodes, n_embedding_nodes, n_hidden_nodes, batch_size=None):

    input_layer = Input(batch_shape=(batch_size,53,300),name='input_layer')
    

    embedding_layer = Embedding(input_dim=n_input_nodes, 
                                output_dim=n_embedding_nodes, 
                                mask_zero=False, name='embedding_layer')(input_layer)

    gru_layer = TimeDistributed(GRU(n_hidden_nodes))(embedding_layer)
    
    flatten_layer = Flatten()(gru_layer)
    
    output_layer = Dense(1, activation="linear", name='output_layer')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(loss="mean_squared_error",
                  optimizer='adam')
    
    return model

rnn_model = create_rnn_model(n_input_nodes=len(train_padded) + 1, n_embedding_nodes=300, n_hidden_nodes=500)
rnn_model.summary()

import numpy as np
from keras.utils import to_categorical

target=df.Sentiment.values


rnn_model.fit(x=train_padded, y=target, epochs=5)



