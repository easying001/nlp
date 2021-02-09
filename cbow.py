# -*- coding: utf-8 -*-
import os
import re
import dataset
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
CORPUS_PATH = "./dataset/sample_test.txt"
PUNCTATION = "?,.？，。！"

def get_vocab_dic():
    file_path = os.path.abspath(CORPUS_PATH)
    vocab = set()
    dataset = []
    print(file_path)
    with open(file_path, mode='r', encoding='utf-8') as fb:
        for line in fb.readlines():
            out = re.sub("[{}]".format(PUNCTATION), '', line)
            out = re.sub("\d*\t", '', out).replace('\n', '')
            dataset.append(out)
            res = list(out.split(" "))
            # print(res)
            vocab.update(res)
    return list(vocab), dataset


VOCAB_DICS, DATASET = get_vocab_dic()
VOCAB_SIZE = len(VOCAB_DICS)
DATASET_I2V = {i:v for i,v in enumerate(VOCAB_DICS)}
DATASET_V2I = {v:i for i,v in enumerate(VOCAB_DICS)}


class CBOW(keras.Model):
    def __init__(self, vocab_dim, emb_dim,):
        super(CBOW, self).__init__()
        self.v_dim = vocab_dim
        self.e_dim = emb_dim
        self.embedding = keras.layers.Embedding(input_dim=vocab_dim,
                                                output_dim=emb_dim,
                                                embeddings_initializer=keras.initializers.RandomNormal(0,0.1))
        self.optimizer = keras.optimizers.Adam(0.01)
        #self.loss = keras.losses.categorical_crossentropy()

    def predict(self, x):
        o = self.embedding(x)
        o = tf.reduce_mean(o, axis=1)
        return o

def train(model, data):
    for i in range(1000):
        x, y = data.sample(8)

def translate(x, y):
    lstx = x.tolist()
    lsty = y.tolist()

    tempx = [DATASET_I2V.get(v) for v in lstx[0]]
    tempy = [DATASET_I2V.get(v) for v in lsty]
    print(tempx, tempy)


if __name__ == '__main__':
    d = dataset.process_w2v_data(DATASET, skip_window=2,method='cbow')
    print("vocab dictionary size = %s" %(VOCAB_SIZE))
    model = CBOW(VOCAB_SIZE, 2)
    x,y = d.sample(1)
    translate(x,y)



