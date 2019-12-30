# _*_coding:utf-8_*_
import pandas as pd
import os
from config import config
import numpy as np
import re
import jieba
from collections import Counter
import pickle

CSV_PREPROCESSED = os.listdir(config.get('preprocessed_path'))

def csv_preprocess():
    '''处理原始数据'''
    preprocessed_path = config.get('preprocessed_path')
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    raw_data_path = config.get('raw_data_path')
    for data_path in os.listdir(raw_data_path):
        items = list()
        labels = list()
        for path, dir_paths, file_paths in os.walk(os.path.join(raw_data_path, data_path)):
            for file_path in file_paths:
                file_path = os.path.join(path, file_path)
                par_labels = re.match(r'.*\\(.*?)_(.*?)\\origin\\(.*?).csv', file_path).groups()
                df = pd.read_csv(file_path)
                items.extend(df.item.apply(lambda x: item_preprocess(x)))
                labels.extend(df.item.apply(lambda x: label_preprocess(x, par_labels)))
        df = pd.DataFrame(zip(items, labels), columns=['items', 'labels'])
        df.to_csv(os.path.join(preprocessed_path, data_path+'.csv'), index=False)

def item_preprocess(text):
    text = text.split('[知识点：]')[0].replace('知识点：', '')\
                                     .replace('[题目]', '').strip()
    text = re.sub('【考点精析】.*', '', text)
    text = re.sub('题型:.*', '', text)
    text = re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+\]\[——！:∶；，。？"
                  r"、~@#%……&*（）·¥\-\|\\《》〈〉～A ① B ② C ③ D ④⑤⑥’‘“”：．﹣→°]", "", text)
    text = ' '.join(jieba.cut(text))
    return text

def label_preprocess(text, par_labels):
    text = list(par_labels) + text.split('[知识点：]')[-1].split(',')
    text = [re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+\]\[——！:∶；，。？"
                  r"、~@#%……&*（）·¥\-\|\\《》〈〉～A ① B ② C ③ D ④⑤⑥’‘“”：．﹣→°]", "", x)
            for x in text]
    return ' '.join(text)

def load_csv():
    history_path = os.path.join(config.get('preprocessed_path'), '高中_历史.csv')
    history = pd.read_csv(history_path)
    items, labels = history['items'], history['labels']
    return items, labels

def generate_vocab():
    history_path = os.path.join(config.get('preprocessed_path'), '高中_历史.csv')
    history = pd.read_csv(history_path)
    items, labels = history['items'], history['labels']
    item_words = list()
    label_words = list()
    for item in items:
        item_words.extend(item.split())
    for label in labels:
        label_words.extend(label.split())
    # baseline用词频代表词的重要性，后期优化可以使用如tf-idf等其他统计表示方法
    item_words = Counter(item_words).most_common(config.get('vocab_size')-1)
    label_words = Counter(label_words)
    word2index = {'UNK': 0}
    index2word = {0: 'UNK'}
    for item_word in item_words:
        word2index[item_word[0]] = len(word2index)
        index2word[len(word2index)] = item_word[0]
    label2index = {'None': 0}
    index2label = {0: 'None'}
    for label_word in label_words:
        label2index[label_word] = len(label2index)
        index2label[len(label2index)] = label_word
    with open(config.get('vocab_path'), 'wb') as f:
        pickle.dump((word2index, index2word, label2index, index2label), f, pickle.HIGHEST_PROTOCOL)
    return word2index, index2word, label2index, index2label

def padding(x, y, word2index, label2index):
    sequence_length = config.get('sequence_length')
    num_classes = config.get('num_classes')
    x_list = list()
    for item in x:
        _item = [word2index.get(i, 0) for i in item.split()]
        x_list.append(_item[:sequence_length] + [0]*(sequence_length-len(_item)))
    y_list = list()
    for item in y:
        _item = [label2index.get(i, 0) for i in item.split()]
        y_list.append(_item[:num_classes] + [0]*(num_classes-len(_item)))
    return np.array(x_list), np.array(y_list)

def shuffle(x, y):
    np.random.seed(config.get('random_state'))
    indices = np.random.permutation(np.arange(len(x)))
    shuffled_x = x[indices]
    shuffled_y = y[indices]
    return shuffled_x, shuffled_y

def split_data(x, y):
    rate = config.get('train_test_dev_rate')
    x, y = shuffle(x, y)
    x_train, y_train = x[:int(len(x)*rate[0])], y[:int(len(x)*rate[0])]
    x_test, y_test = x[int(len(x)*rate[0]): int(len(x)*(rate[0]+rate[1]))],\
                     y[int(len(x)*rate[0]): int(len(x)*(rate[0]+rate[1]))]
    x_val, y_val = x[int(len(x)*(rate[0]+rate[1])):], y[int(len(x)*(rate[0]+rate[1]))]
    return x_train, y_train, x_test, y_test, x_val, y_val

def generate_batches(x, y, shuffle=True):
    batch_size = config.get('batch_size')
    data = np.array(list(zip(x, y)))
    nums_per_epoch = len(data) // batch_size
    for epoch in range(config.get('num_epochs')):
        if shuffle:
            np.random.seed(config.get('random_state'))
            shuffle_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffle_indices]
        for batch in range(nums_per_epoch):
            start_index = batch*batch_size
            end_index = min(len(data), batch_size*(batch+1))
            yield data[start_index: end_index]

if __name__ == "__main__":
    if not CSV_PREPROCESSED:
        csv_preprocess()
    X, y = load_csv()
    word2index, index2word, label2index, index2label = generate_vocab()
    input_x, input_y = padding(X, y, word2index, label2index)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(input_x, input_y)
    batches = generate_batches(x_train, y_train)
    print()