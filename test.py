import tensorflow as tf

cce = tf.keras.losses.CategoricalCrossentropy()

indices = [[0,1], [1,2], [0,2]]
depth = 3
output = tf.reduce_sum(tf.one_hot(indices, depth), axis=1)
#
# t = tf.keras.losses.categorical_crossentropy(output, tf.convert_to_tensor([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
#
# loss = cce(
#   output,
#   tf.convert_to_tensor([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
# print('Loss: ', loss.numpy())

s = tf.keras.metrics.CosineSimilarity(axis=0)
s.update_state(output, tf.convert_to_tensor([[.7, .25, .05], [.05, .79, .16], [.15, .11, .74]]))
print(s.result())

ss = tf.keras.metrics.CategoricalAccuracy()
ss.update_state(output, tf.convert_to_tensor([[.7, .25, .05], [.05, .79, .16], [.15, .11, .74]]))
print(ss.result())