# -*- coding: utf-8 -*-
import os
import re
import string
# import tensorflow as tf
import numpy as np

CORPUS_PATH = "./dataset/sample_test.txt"
PUNCTATION = "?,.？，。"


def process_data():
    file_path = os.path.abspath(CORPUS_PATH)

    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as fb:
        for line in fb.readlines():
            out = re.sub("[{}]".format(PUNCTATION), '', line)
            out = re.sub("\d*\t", '', out).replace('\n','')
            res = list(out.split(" "))
            print(res)

if __name__ == '__main__':
    process_data()