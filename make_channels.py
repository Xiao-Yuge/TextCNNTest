# _*_coding:utf-8_*_
# 使用fasttext、word2vec、glove训练三种不同的词向量，
# 使用三种向量表征的文本模拟三个通道进行卷积操作
# 1、从preprocessed_data/vocabs.dict中读取index2word
# 2、训练词向量生成index2vec
# 3、使用np.array(index2vec.values())作为预训练matrix传到模型中，在Embedding层生效

from config import config
import numpy as np
import gensim

def train_fasttext():
    pass

def train_word2vec():
    pass

def train_glove():
    pass

def make_fasttext_channel():
    pass

def make_word2vec_channel():
    pass

def make_glove_channel():
    pass