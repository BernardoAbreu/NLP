#!/usr/bin/env python3
import numpy as np
import pandas as pd
# import re
import gensim
# import nltk
import keras
# import matplotlib.pyplot as plt
from keras.models import model_from_json


# In[2]:

paths = {
    'train': '../macmorpho-v3/macmorpho-train.txt',
    'test': '../macmorpho-v3/macmorpho-test.txt',
    'dev': '../macmorpho-v3/macmorpho-dev.txt',
    'word2vec': '../data/skip_s50.txt'
}


# ## Embedding - Word2Vec
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(paths['word2vec'])
vocab_size, embedding_size = w2v_model.vectors.shape

# ### Adiciona vetores extras
w2v_model.add(['<PAD>', '<OOV>'],
              [[0.1] * embedding_size,
              [0.2] * embedding_size])


# ## Leitura do texto
def read_text(filename):
    with open(filename, 'r') as f:
        return f.readlines()


train_text = read_text(paths['train'])
test_text = read_text(paths['test'])
dev_text = read_text(paths['dev'])


# ### Separação de palavras e tags
def split_word_tags(text):
    word_lines = []
    tag_lines = []
    for line in text:
        words, tags = zip(*[tagged_word.split('_')
                            for tagged_word in line.split()])
        word_lines.append([w.lower() for w in words])
        tag_lines.append(list(tags))
    return word_lines, tag_lines


train_words, train_tags = split_word_tags(train_text)
print(train_words[0])
print(train_tags[0])

test_words, test_tags = split_word_tags(test_text)
dev_words, dev_tags = split_word_tags(dev_text)


def flat_list(l):
    return [item for sublist in l for item in sublist]

id2tag = ['<PAD>'] + list(set(flat_list(train_tags)).union(set(flat_list(test_tags))).union(set(flat_list(dev_tags))))
tag2id = {}
for i, tag in enumerate(id2tag):
    tag2id[tag] = i

# ## Pad the words

# ### Analyse sentence size distribution
df_train = pd.DataFrame(columns=['words', 'tags'])
df_test = pd.DataFrame(columns=['words', 'tags'])
df_dev = pd.DataFrame(columns=['words', 'tags'])

df_train['words'] = train_words
df_train['tags'] = train_tags

df_test['words'] = test_words
df_test['tags'] = test_tags

df_dev['words'] = dev_words
df_dev['tags'] = dev_tags


df_sentences = pd.concat([df_train, df_test, df_dev], axis=0)

MAX_SENTENCE_LENGTH = int(df_sentences['words'].map(len).describe()['75%'])


def fill_sentence(sentence):
    tokens_to_fill = int(MAX_SENTENCE_LENGTH - len(sentence))
    sentence.extend(['<PAD>'] * tokens_to_fill)
    return sentence[:MAX_SENTENCE_LENGTH]


df_test["words"] = df_test["words"].map(fill_sentence)
df_test["tags"] = df_test["tags"].map(fill_sentence)

print(len(w2v_model.vocab))
print(MAX_SENTENCE_LENGTH)
print(len(df_train))
w2v_model.vocab['<OOV>'].index
print(len(df_train['words']))


pretrained_weights = w2v_model.vectors
vocab_size, embedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)


def word2idx(word):
    return w2v_model.vocab[word].index


def idx2word(idx):
    return w2v_model.index2word[idx]


def prepare_words(sentences):
    sentences_x = np.zeros([len(sentences), MAX_SENTENCE_LENGTH],
                           dtype=np.int32)

    oov_index = word2idx('<OOV>')
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            try:
                sentences_x[i, t] = word2idx(word)
            except KeyError:
                sentences_x[i, t] = oov_index
    return sentences_x


def prepare_tags(tag_sentences, tag2index):
    tags_y = np.zeros([len(tag_sentences), MAX_SENTENCE_LENGTH],
                      dtype=np.int32)
    for i, sentence in enumerate(tag_sentences):
        for t, tag in enumerate(sentence):
            tags_y[i, t] = tag2index[tag]
    return tags_y


print('\nPreparing the test data for LSTM...')
test_sentences_X = prepare_words(df_test['words'])
print('test_x shape:', test_sentences_X.shape)

print('\nPreparing the test data for LSTM...')
test_tags_y = prepare_tags(df_test['tags'], tag2id)
print('test_y shape:', test_tags_y.shape)

print()

cat_test_tags_y = keras.utils.to_categorical(test_tags_y,
                                             num_classes=len(id2tag),
                                             dtype='int32')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(0.001),
                     metrics=['accuracy'])

score = loaded_model.evaluate(test_sentences_X, cat_test_tags_y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
