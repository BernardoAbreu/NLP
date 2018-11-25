#!/usr/bin/env python3
# coding: utf-8

# # Trabalho Prático 2
# ## Processamento de Linguagem Natural - 2018/2
# ### Bernardo de Almeida Abreu - 2018718155

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from utils import *
from sklearn.metrics import classification_report


def main():
    train_sentences_X = np.loadtxt('train_x')
    train_tags_y = np.loadtxt('train_y')
    test_sentences_X = np.loadtxt('test_x')
    test_tags_y = np.loadtxt('test_y')
    dev_sentences_X = np.loadtxt('dev_x')
    dev_tags_y = np.loadtxt('dev_y')
    cat_train_tags_y = keras.utils.to_categorical(train_tags_y)
    cat_test_tags_y = keras.utils.to_categorical(test_tags_y)
    cat_dev_tags_y = keras.utils.to_categorical(dev_tags_y)
    print(test_sentences_X.shape)
    model = keras.models.load_model('keras_model.hdf5',
                                    custom_objects={'ignore_accuracy':
                                                    ignore_class_accuracy(0)})
    saver = tf.train.Saver()
    sess = keras.backend.get_session()
    saver.restore(sess, './keras_model')
    model.summary()
    # evaluate_all(model,
    #              train_sentences_X, cat_train_tags_y,
    #              test_sentences_X, cat_test_tags_y,
    #              dev_sentences_X, cat_dev_tags_y)
    scores = []
    scores.append(model.evaluate(train_sentences_X, cat_train_tags_y))
    scores.append(model.evaluate(test_sentences_X, cat_test_tags_y))
    scores.append(model.evaluate(dev_sentences_X, cat_dev_tags_y))
    np.savetxt('scores.txt', scores, header=','.join(model.metrics_names),
               delimiter=',')

    test_pred = model.predict_classes(test_sentences_X)
    np.savetxt('test_predict.txt', test_pred, fmt='%s')
    train_pred = model.predict_classes(train_sentences_X)
    np.savetxt('train_predict.txt', train_pred, fmt='%s')
    dev_pred = model.predict_classes(dev_sentences_X)
    np.savetxt('dev_predict.txt', dev_pred, fmt='%s')
    # for x in test_sentences_X:
    #     # if x.shape != (25,):
    #     print(x.shape)
    #     inputs.append(model.predict(x))
    # np.savetxt('tt.txt', inputs)


if __name__ == '__main__':
    main()
