import numpy as np
import gensim
import pickle
# import tensorflow.contrib.keras as keras
import keras


def load_embedding(path, load_pickle=False):
    if load_pickle:
        m = pickle.load(open('word2vec_model_skipgram_100.p', 'rb'))
    else:
        m = gensim.models.KeyedVectors.load_word2vec_format(path)
        embed_size = m.vectors.shape[1]
        # ### Adiciona vetores extras
        m.add(['<PAD>', '<OOV>'], [[0.1] * embed_size, [0.2] * embed_size])
        pickle.dump(m, open('word2vec_model_skipgram_100.p', 'wb'))
    return m


# ## Leitura do texto
def read_text(filename):
    with open(filename, 'r') as f:
        return f.readlines()


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


def flat_list(l):
    return [item for sublist in l for item in sublist]


def fill_sentence(sentence, max_sentence_length):
    tokens_to_fill = int(max_sentence_length - len(sentence))
    sentence.extend(['<PAD>'] * tokens_to_fill)
    return sentence[:max_sentence_length]


def idx2word(idx, w2v_model):
    return w2v_model.index2word[idx]


def prepare_words(sentences, w2v_model, max_sentence_length):
    sentences_x = np.zeros([len(sentences), max_sentence_length],
                           dtype=np.int32)

    oov_index = w2v_model.vocab['<OOV>'].index
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            try:
                sentences_x[i, t] = w2v_model.vocab[word].index
            except KeyError:
                sentences_x[i, t] = oov_index
    return sentences_x


def prepare_tags(tag_sentences, tag2index, max_sentence_length):
    tags_y = np.zeros([len(tag_sentences), max_sentence_length],
                      dtype=np.int32)
    for i, sentence in enumerate(tag_sentences):
        for t, tag in enumerate(sentence):
            tags_y[i, t] = tag2index[tag]
    return tags_y


def create_architecture(w2v_model, input_len, output_len, lstm_size=256):
    model = keras.models.Sequential()

    vocab_size, embedding_size = w2v_model.vectors.shape
    # ### Adiciona camada de embedding
    model.add(
        keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            input_length=input_len,
            weights=[w2v_model.vectors]
        )
    )

    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_size, return_sequences=True)
        )
    )

    # model.add(keras.layers.Dropout(0.2))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(output_len)
        )
    )

    model.add(keras.layers.Activation('softmax'))
    return model


def evaluate_all(model, train_X, train_Y, test_X, test_Y, dev_X, dev_Y):
    print('Evaluating model:')
    scores = model.evaluate(test_X, test_Y)
    for metric, score in zip(model.metrics_names, scores):
        print(f"Test model {metric}: {score*100}")

    scores = model.evaluate(train_X, train_Y)
    for metric, score in zip(model.metrics_names, scores):
        print(f"Train model {metric}: {score*100}")

    scores = model.evaluate(dev_X, dev_Y)
    for metric, score in zip(model.metrics_names, scores):
        print(f"Dev model {metric}: {score*100}")


def prepare_data(df, w2v_model, tag2id, max_sentence_len):
    x_data = prepare_words(df['words'], w2v_model, max_sentence_len)
    y_data = prepare_tags(df['tags'], tag2id, max_sentence_len)
    return x_data, y_data


def ignore_accuracy(y_true, y_pred):
    y_true_class = keras.backend.argmax(y_true, axis=-1)
    y_pred_class = keras.backend.argmax(y_pred, axis=-1)
    ignore_mask = keras.backend.cast(
        keras.backend.not_equal(y_pred_class, 0), 'int32')
    matches = keras.backend.cast(
        keras.backend.equal(y_true_class, y_pred_class), 'int32') * \
        ignore_mask
    accuracy = keras.backend.sum(matches) / \
        keras.backend.maximum(keras.backend.sum(ignore_mask), 1)
    return accuracy


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = keras.backend.argmax(y_true, axis=-1)
        y_pred_class = keras.backend.argmax(y_pred, axis=-1)
        ignore_mask = keras.backend.cast(
            keras.backend.not_equal(y_pred_class, to_ignore), 'int32')
        matches = keras.backend.cast(
            keras.backend.equal(y_true_class, y_pred_class), 'int32') * \
            ignore_mask
        accuracy = keras.backend.sum(matches) / \
            keras.backend.maximum(keras.backend.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy
