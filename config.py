# _*_coding:utf-8_*_
# 配置一些训练过程中使用到的参数

config = {
    'sequence_length': 128,  # 最大文本长度
    'num_classes': 5,  # 最大分类数
    'all_classes': 10,  # 样本中总分类数（训练过程修改）
    'vocab_size': 5000,  # 词表大小
    'embedding_size': 512,  # 词向量大小
    'device': '/cpu:0',  # 设备：默认为cpu
    'batch_size': 64,  # batch大小
    'num_epochs': 10,  # 训练epoch个数
    'evaluate_every': 100,  # 多少个batch验证一次
    'checkpoint_every': 100,  # 多少个batch保存一次
    'num_checkpoint': 5,  # 最大checkpoints个数
    'allow_soft_placement': True,  # 是否允许程序自动选择备用device
    'log_device_placement': True,  # 是否允许在终端打印日志文件
    'train_test_dev_rate': [0.7, 0.2, 0.1],  # 训练集、测试集、验证集
    'learning_rate': 0.001,  # 学习率
    'raw_data_path': './data',  # 数据路径
    'preprocessed_path': './preprocessed_data',  # 数据预处理保存路径
    'vocab_path': './preprocessed_data/vocabs.dict',  # 词典路径
    'checkpoint_path': './checkpoint',  # 模型保存路径
    'log_path': './logs.txt',  # 日志保存路径
    'random_state': 232,  # 随机seed
    'channel_nums': 1,  # 通道个数，可以使用word2vec, fasttext, glove生成的词向量模拟三个通道
    'filter_size': [3, 4, 5],  # 卷积核长度（宽度为embedding_size）
    'regularize_lambda': 0.001,  # 正则化惩罚项λ
    'dropout_rate': 0.3  # dropout rate
}