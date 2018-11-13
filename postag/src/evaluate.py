#!/usr/bin/env python3
# coding: utf-8

# # Trabalho Prático 2
# ## Processamento de Linguagem Natural - 2018/2
# ### Bernardo de Almeida Abreu - 2018718155

import pandas as pd
import keras
import tensorflow as tf
from utils import *


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

    model = keras.models.load_model('keras_model.hdf5')
    saver = tf.train.Saver()
    sess = keras.backend.get_session()
    saver.restore(sess, './keras_model')
    model.summary()
    evaluate_all(model,
                 train_sentences_X, cat_train_tags_y,
                 test_sentences_X, cat_test_tags_y,
                 dev_sentences_X, cat_dev_tags_y)


if __name__ == '__main__':
    main()
