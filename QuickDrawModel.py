"""
A slightly modified VGG Network
"""
from AuxiliaryCNN import csv_generator, text_to_labels, get_batch, conv_layer
import numpy as np
import tensorflow as tf
from subprocess import check_output

# constants
DIRPATH = "/data/scratch/epeake/Google-Doodles/"
BATCH_SIZE = 30
HEIGHT = 256
WIDTH = 256
N_EPOCHS = 5
LEARNING_RATE = 0.0003

label_to_class = text_to_labels(DIRPATH)
class_eye = np.eye(len(label_to_class))
n_outputs = len(label_to_class)
csv_len = int(check_output('wc -l ' + DIRPATH + 'train.csv | grep -o "[0-9]\+"', shell=True))


with tf.device("/gpu:1"):
    X = tf.placeholder("float", [None, HEIGHT, WIDTH, 1])   # [None, HEIGHT, width, channels]
    Y = tf.placeholder("float", [None, n_outputs])       # [None, classes]

    conv_1_1 = conv_layer(X, 1, 64, "conv_1_1")
    conv_1_2 = conv_layer(conv_1_1, 64, 64, "conv_1_2")
    pool1 = tf.nn.max_pool(conv_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    conv_2_1 = conv_layer(pool1, 64, 128, "conv_2_1")
    conv_2_2 = conv_layer(conv_2_1, 128, 128, "conv_2_2")
    pool2 = tf.nn.max_pool(conv_2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='Same', name="pool2")

    conv_3_1 = conv_layer(pool2, 128, 256, "conv_3_1")
    conv_3_2 = conv_layer(conv_3_1, 256, 256, "conv_3_2")
    conv_3_3 = conv_layer(conv_3_2, 256, 256, "conv_3_3")
    pool3 = tf.nn.max_pool(conv_3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

    conv_4_1 = conv_layer(pool3, 256, 512, "conv_4_1")
    conv_4_2 = conv_layer(conv_4_1, 512, 512, "conv_4_2")
    conv_4_3 = conv_layer(conv_4_2, 512, 512, "conv_4_3")
    pool4 = tf.nn.max_pool(conv_4_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

    conv_5_1 = conv_layer(pool4, 512, 512, "conv_5_1")
    conv_5_2 = conv_layer(conv_5_1, 512, 512, "conv_5_2")
    conv_5_3 = conv_layer(conv_5_2, 512, 512, "conv_5_3")
    pool5 = tf.nn.max_pool(conv_5_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")

    pre_fully_connected = tf.contrib.layers.flatten(pool5, scope="flattened")

    fully_connected_1 = tf.layers.dense(pre_fully_connected, 410,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.nn.relu, name="fc1")

    fully_connected_2 = tf.layers.dense(fully_connected_1, 410,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.nn.relu, name="fc2")

    logits = tf.layers.dense(fully_connected_2, n_outputs, activation=tf.nn.relu, name="logits")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, name="softmax"))

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        n_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))


config = tf.ConfigProto()
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    batch_number = 1
    total_correct = 0
    for epoch in range(N_EPOCHS):
        csv_gen = csv_generator(DIRPATH, BATCH_SIZE)
        step_i = 0
        while True:
            try:
                X_batch, Y_batch = get_batch(csv_gen, label_to_class, class_eye)
            except StopIteration:
                break
            sess.run(train_step, feed_dict={X: X_batch, Y: Y_batch})
            if step_i % 500 == 0:
                total_correct += n_correct.eval({X: X_batch, Y: Y_batch})
                train_accuracy = total_correct / (batch_number * BATCH_SIZE)
                print("Epoch:", epoch + 1, "Total Batch Number:", batch_number, "Train accuracy:", train_accuracy)
                saver.save(sess, "./qd_model/quick_draw_model", step_i)
            batch_number += 1
            step_i += 1

    print("Total accuracy: ", total_correct / (csv_len * N_EPOCHS))
    save_path = saver.save(sess, "./qd_model/quick_draw_model")
