# _*_coding:utf-8_*_
# 使用fasttext、word2vec、glove训练三种不同的词向量，
# 使用三种向量表征的文本模拟三个通道进行卷积操作
# 1、从preprocessed_data/vocabs.dict中读取index2word
# 2、训练词向量生成index2vec
# 3、使用np.array(index2vec.values())作为预训练matrix传到模型中，在Embedding层生效

from config import config
import numpy as np
from gensim.models import FastText, Word2Vec, KeyedVectors
import pandas as pd
import os
import pickle

def train_fasttext(documents, save_dir):
    # https://blog.csdn.net/sinat_26917383/article/details/83041424+
    if not os.path.exists(save_dir):
        embedding_size = config.get('embedding_size')
        model = FastText(documents, size=embedding_size, window=3, hs=1,
                         min_count=1, iter=10, min_n=3, max_n=20, word_ngrams=1)
        model.wv.save_word2vec_format(save_dir, binary=True)

def train_word2vec(documents, save_dir):
    if not os.path.exists(save_dir):
        embedding_size = config.get('embedding_size')
        model = Word2Vec(documents, size=embedding_size, window=3, hs=1,
                         min_count=1, iter=10)
        model.wv.save_word2vec_format(save_dir, binary=True)

def make_channel(word2index, save_dir):
    models = KeyedVectors.load_word2vec_format(save_dir, binary=True)
    matrix = list()
    for word in word2index.keys():
        if word in models.index2word:
            matrix.append(models.vectors[models.index2word.index(word)])
        else:
            matrix.append(np.random.uniform(-1, 1, config.get('embedding_size')))
    return np.array(matrix)

def load_vocabs():
    vocab_path = config.get('vocab_path')
    with open(vocab_path, 'rb') as fr:
        word2index, index2word, label2index, index2label = pickle.load(fr)
    return word2index, index2word, label2index, index2label

def get_data():
    preprocessed_data_dir = config.get('preprocessed_path')
    documents = list()
    for path, dirs, files in os.walk(preprocessed_data_dir):
        for file in files:
            if file.endswith('csv'):
                csv = pd.read_csv(os.path.join(path, file))
                documents.extend([str(item).split() for item in csv['items'].to_list()])
    return documents

def make_channels():
    word2index, index2word, label2index, index2label = load_vocabs()
    documents = get_data()
    model_path = config.get('preprocessed_path')
    fasttext_model_path = os.path.join(model_path, 'fast_text.model')
    w2v_model_path = os.path.join(model_path, 'word2vec.model')

    train_fasttext(documents, fasttext_model_path)
    train_word2vec(documents, w2v_model_path)

    fasttext_matrix = make_channel(word2index, fasttext_model_path)
    w2v_matrix = make_channel(word2index, w2v_model_path)
    return np.array([fasttext_matrix, w2v_matrix])

if __name__ == "__main__":
    channels = make_channels()