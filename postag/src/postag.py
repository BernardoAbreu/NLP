#!/usr/bin/env python3
# coding: utf-8

# # Trabalho Prático 2
# ## Processamento de Linguagem Natural - 2018/2
# ### Bernardo de Almeida Abreu - 2018718155

import pandas as pd
import tensorflow as tf
import keras
import pickle
from utils import *


def main():
    train_text = read_text(PATHS['train'])
    test_text = read_text(PATHS['test'])
    dev_text = read_text(PATHS['dev'])

    train_words, train_tags = split_word_tags(train_text)
    test_words, test_tags = split_word_tags(test_text)
    dev_words, dev_tags = split_word_tags(dev_text)

    id2tag = ['<PAD>'] + list(set(flat_list(train_tags))
                              .union(set(flat_list(test_tags)))
                              .union(set(flat_list(dev_tags))))
    tag2id = {tag: i for i, tag in enumerate(id2tag)}

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
    max_sentence_len = int(df_sentences['words'].map(len).describe()['75%'])

    f_lambda = lambda x: fill_sentence(x, max_sentence_len)
    df_train["words"] = df_train["words"].map(f_lambda)
    df_train["tags"] = df_train["tags"].map(f_lambda)

    df_test["words"] = df_test["words"].map(f_lambda)
    df_test["tags"] = df_test["tags"].map(f_lambda)

    df_dev["words"] = df_dev["words"].map(f_lambda)
    df_dev["tags"] = df_dev["tags"].map(f_lambda)

    w2v_model = load_embedding(True)

    print('\nPreparing the train data for LSTM...')
    train_sentences_X, train_tags_y = prepare_data(df_train, w2v_model,
                                                   tag2id, max_sentence_len)

    print('\nPreparing the test data for LSTM...')
    test_sentences_X, test_tags_y = prepare_data(df_test, w2v_model,
                                                 tag2id, max_sentence_len)

    print('\nPreparing the validation data for LSTM...')
    dev_sentences_X, dev_tags_y = prepare_data(df_dev, w2v_model,
                                               tag2id, max_sentence_len)

    np.savetxt('train_x', train_sentences_X)
    np.savetxt('test_x', test_sentences_X)
    np.savetxt('dev_x', dev_sentences_X)
    np.savetxt('train_y', train_tags_y)
    np.savetxt('test_y', test_tags_y)
    np.savetxt('dev_y', dev_tags_y)
    print()

    cat_train_tags_y = keras.utils.to_categorical(train_tags_y,
                                                  num_classes=len(id2tag))
    cat_test_tags_y = keras.utils.to_categorical(test_tags_y,
                                                 num_classes=len(id2tag))
    cat_dev_tags_y = keras.utils.to_categorical(dev_tags_y,
                                                num_classes=len(id2tag))

    # ## Arquitetura do modelo
    print('Creating model')
    model = create_architecture(w2v_model, max_sentence_len, len(tag2id))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model.summary()

    csv_logger = keras.callbacks.CSVLogger('training.log')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.001,
                                               patience=4,
                                               verbose=1,
                                               mode='min')
    model.fit(train_sentences_X, cat_train_tags_y,
              batch_size=64, epochs=1,
              validation_data=(dev_sentences_X, cat_dev_tags_y),
              callbacks=[csv_logger, early_stop])

    saver = tf.train.Saver()
    sess = keras.backend.get_session()
    saver.save(sess, './keras_model')
    model.save('keras_model.hdf5')
    evaluate_all(model,
                 train_sentences_X, cat_train_tags_y,
                 test_sentences_X, cat_test_tags_y,
                 dev_sentences_X, cat_dev_tags_y)

    # ## Save model
    # ### serialize weights to HDF5
    # model.save_weights("model_weights.hdf5", overwrite=True)
    # print("Saved model to disk")
    # # Architecture
    # with open("model.json", "w") as f:
    #     f.write(model.to_json())
    # w = model.get_weights()
    # print(w.shape)
    # np.savetxt('weights.csv', w, fmt='%s')


if __name__ == '__main__':
    main()
