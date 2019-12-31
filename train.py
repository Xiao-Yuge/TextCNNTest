# _*_coding:utf-8_*_
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from data_preprocesser import *
from config import config
from make_channels import make_channels
from text_cnn import TextCNN

def validate(X, y, model, loss_object, accuracy_object):
    output = model(X)
    one_hot = tf.reduce_sum(tf.one_hot(y, config.get('all_classes')), axis=1)
    loss = loss_object(one_hot, output)
    accuracy_object.update_state(one_hot, output)
    accuracy = accuracy_object.result()
    return loss, accuracy

def operate_logs(ifconsole, logs):
    log_path = config.get('log_path')
    with open(log_path, 'a+') as fw:
        fw.write(logs + '\n')
    if ifconsole:
        print(logs)

@tf.function
def train_step(X, y, model, loss_object, optimizer, accuracy_object):
    with tf.GradientTape() as tape:
        output = model(X)
        one_hot = tf.reduce_sum(tf.one_hot(y, config.get('all_classes')), axis=1)
        loss = loss_object(one_hot, output)
        accuracy_object.update_state(one_hot, output)
        accuracy = accuracy_object.result()
        variables = model.trainable_variables
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))
        return loss, accuracy

def train():
    if not CSV_PREPROCESSED:
        csv_preprocess()
    X, y = load_csv()
    word2index, index2word, label2index, index2label = generate_vocab()
    input_x, input_y = padding(X, y, word2index, label2index)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(input_x, input_y)
    batches = generate_batches(x_train, y_train)

    initial_lr = config.get('learning_rate')
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=1000,
        decay_rate=0.8,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    accuracy_object = tf.keras.metrics.CategoricalAccuracy()
    initial_matrix = make_channels()
    text_cnn = TextCNN(initial_matrix)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=text_cnn)
    manager = tf.train.CheckpointManager(checkpoint,
                                         config.get('checkpoint_path'),
                                         max_to_keep=config.get('num_checkpoint'))
    loss, accuracy = 0, 0
    operate_logs(True, "--"*25+"training started"+"--"*25)
    for i, batch in enumerate(batches):
        X, Y = np.array(batch[0]), np.array(batch[1])
        if (i+1) % config.get('evaluate_every') == 0:
            evaluate_loss, evaluate_accuracy = validate(x_val, y_val, text_cnn, loss_object, accuracy_object)
            operate_logs(config.get('log_device_placement'),
                        "{}:  step {}, train loss:{:.2f}, train accuracy:{:.2f}, "
                        "evaluate loss:{:.2f}, evaluate accuracy:{:.2f}"\
                        .format(str(time.time()), str(i), loss/config.get('evaluate_every'),
                                accuracy/config.get('evaluate_every'), evaluate_loss, evaluate_accuracy))
            loss, accuracy = 0, 0
        if (i+1) % config.get('checkpoint_every') == 0:
            manager.save()
            operate_logs(config.get('log_device_placement'),
                        "{}: step {}, checkpoint file saved." \
                        .format(str(time.time()), str(i)))
        batch_loss, batch_accuracy = train_step(X, Y, text_cnn, loss_object, optimizer, accuracy_object)
        loss += batch_loss
        accuracy += batch_accuracy
        operate_logs(False,
                     "step {}, train loss:{:.2f}, train accuracy:{:.2f}"
                     .format(str(i), batch_loss, batch_accuracy))
    operate_logs(True, "--" * 25 + "training finished" + "--" * 25)

if __name__ == "__main__":
    train()