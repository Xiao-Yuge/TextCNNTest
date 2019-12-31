import tensorflow as tf
from tensorflow import keras
from config import config
from make_channels import make_channels
import numpy as np

class TextCNN(keras.models.Model):
    def __init__(self, matrixes=None):
        super(TextCNN, self).__init__()
        self.vocab_size = config.get('vocab_size')
        self.embedding_size = config.get('embedding_size')
        self.sequence_length = config.get('sequence_length')
        self.filter_size = config.get('filter_size')
        num_classes = config.get('all_classes')
        regularize_lambda = config.get('regularize_lambda')
        dropout_rate = config.get('dropout_rate')

        self.matrixes = matrixes

        self.embeddings = list()
        self.conv2ds_pools = list()
        self.flatten = keras.layers.Flatten(data_format='channels_last')
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.dense = keras.layers.Dense(units=num_classes, activation='sigmoid',
                                        kernel_initializer=keras.initializers.constant(0.1),
                                        kernel_regularizer=keras.regularizers.l2(regularize_lambda),
                                        bias_regularizer=keras.regularizers.l2(regularize_lambda))

        # 生成多通道embedding
        channels = 0
        if self.matrixes is None:
            embedding = keras.layers.Embedding(self.vocab_size, self.embedding_size,
                                                embeddings_initializer=keras.initializers.glorot_uniform,
                                                input_length=self.sequence_length)
            channels += 1
            self.embeddings.append(embedding)
        else:
            for matrix in self.matrixes:
                embedding = keras.layers.Embedding(self.vocab_size, self.embedding_size,
                                                   embeddings_initializer=tf.constant_initializer(np.array(matrix)),
                                                   trainable=False)
                channels += 1
                self.embeddings.append(embedding)
        # 生成多通道多size卷积核与池化层
        for filter_size in self.filter_size:
            conv2d = keras.layers.Conv2D(filters=channels, kernel_size=(filter_size, self.embedding_size),
                                         strides=(1, 1), padding='valid', data_format='channels_last',
                                         activation='relu',kernel_initializer='glorot_uniform',
                                         bias_initializer=keras.initializers.constant(0.1))
            pool = keras.layers.MaxPool2D(pool_size=(self.sequence_length - filter_size + 1, 1), strides=(1, 1),
                                          padding='valid', data_format='channels_last')
            self.conv2ds_pools.append([conv2d, pool])


    def call(self, inputs):
        embeddings = list()
        for embedding in self.embeddings:
            expand_embedding = tf.expand_dims(embedding(inputs), axis=-1)
            embeddings.append(expand_embedding)
        embeddings = keras.layers.concatenate(embeddings, axis=-1)
        pool_outputs = list()
        for cov, pool in self.conv2ds_pools:
            cov_output = cov(embeddings)
            pool_output = pool(cov_output)
            pool_outputs.append(pool_output)
        concat_output = keras.layers.concatenate(pool_outputs, axis=-1)
        flatten_output = self.flatten(concat_output)
        droutput_output = self.dropout(flatten_output)
        output = self.dense(droutput_output)
        return output


if __name__ == "__main__":
    import numpy as np
    batch_sz = config.get('batch_size')
    sequence_length = config.get('sequence_length')
    vocab_size = config.get('vocab_size')
    embedding_size = config.get('embedding_size')
    inputs = np.random.random((batch_sz, sequence_length))
    text_cnn = TextCNN(make_channels())
    output = text_cnn(inputs)