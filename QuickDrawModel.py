"""
A slightly modified VGG Network
"""
from AuxiliaryCNN import csv_generator, text_to_labels, get_batch
import numpy as np
import tensorflow as tf
from subprocess import check_output
from re import search

# constants
DIRPATH = "/data/scratch/epeake/Google-Doodles/"
BATCH_SIZE = 100
HEIGHT = 256
WIDTH = 256
N_EPOCHS = 1
LEARNING_RATE = 0.00006

label_to_class = text_to_labels(DIRPATH)
class_eye = np.eye(len(label_to_class))
n_outputs = len(label_to_class)
csv_len = str(check_output("wc -l " + DIRPATH + "all_doodles.csv", shell=True))
csv_len = int(search("[0-9]+", csv_len).group())


with tf.device("/gpu:1"):
    X = tf.placeholder("float", [None, HEIGHT, WIDTH, 1])   # [None, HEIGHT, width, channels]
    Y = tf.placeholder("float", [None, n_outputs])       # [None, classes]

    W_1_1 = tf.get_variable("W_1_1", [3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv_1_1 = tf.nn.conv2d(X, W_1_1, strides=[1, 1, 1, 1], padding="SAME", name="conv_1_1")
    active_1_1 = tf.nn.relu(conv_1_1)
    W_1_2 = tf.get_variable("W_1_2", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv_1_2 = tf.nn.conv2d(active_1_1, W_1_2, strides=[1, 1, 1, 1], padding="SAME", name="conv_1_2")
    active_1_2 = tf.nn.relu(conv_1_2)
    pool1 = tf.nn.max_pool(active_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    W_2_1 = tf.get_variable("W_2_1", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv_2_1 = tf.nn.conv2d(pool1, W_2_1, strides=[1, 1, 1, 1], padding="SAME", name="conv_2_1")
    active_2_1 = tf.nn.relu(conv_2_1)
    W_2_2 = tf.get_variable("W_2_2", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv_2_2 = tf.nn.conv2d(active_2_1, W_2_2, strides=[1, 1, 1, 1], padding="SAME", name="conv_2_2")
    active_2_2 = tf.nn.relu(conv_2_2)
    pool2 = tf.nn.max_pool(active_2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")

    W_3_1 = tf.get_variable("W_3_1", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv_3_1 = tf.nn.conv2d(pool2, W_3_1, strides=[1, 1, 1, 1], padding="SAME", name="conv_3_1")
    active_3_1 = tf.nn.relu(conv_3_1)
    W_3_2 = tf.get_variable("W_3_2", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv_3_2 = tf.nn.conv2d(active_3_1, W_3_2, strides=[1, 1, 1, 1], padding="SAME", name="conv_3_2")
    active_3_2 = tf.nn.relu(conv_3_2)
    W_3_3 = tf.get_variable("W_3_3", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv_3_3 = tf.nn.conv2d(active_3_2, W_3_3, strides=[1, 1, 1, 1], padding="SAME", name="conv_3_3")
    active_3_3 = tf.nn.relu(conv_3_3)
    pool3 = tf.nn.max_pool(active_3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

    W_4_1 = tf.get_variable("W_4_1", [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv_4_1 = tf.nn.conv2d(pool3, W_4_1, strides=[1, 1, 1, 1], padding="SAME", name="conv_4_1")
    active_4_1 = tf.nn.relu(conv_4_1)
    W_4_2 = tf.get_variable("W_4_2", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv_4_2 = tf.nn.conv2d(active_4_1, W_4_2, strides=[1, 1, 1, 1], padding="SAME", name="conv_4_2")
    active_4_2 = tf.nn.relu(conv_4_2)
    W_4_3 = tf.get_variable("W_4_3", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv_4_3 = tf.nn.conv2d(active_4_2, W_4_3, strides=[1, 1, 1, 1], padding="SAME", name="conv_4_3")
    active_4_3 = tf.nn.relu(conv_4_3)
    pool4 = tf.nn.max_pool(active_4_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

    W_5_1 = tf.get_variable("W_5_1", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv_5_1 = tf.nn.conv2d(pool4, W_5_1, strides=[1, 1, 1, 1], padding="SAME", name="conv_5_1")
    active_5_1 = tf.nn.relu(conv_5_1)
    W_5_2 = tf.get_variable("W_5_2", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv_5_2 = tf.nn.conv2d(active_5_1, W_5_2, strides=[1, 1, 1, 1], padding="SAME", name="conv_5_2")
    active_5_2 = tf.nn.relu(conv_5_2)
    W_5_3 = tf.get_variable("W_5_3", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv_5_3 = tf.nn.conv2d(active_5_2, W_5_3, strides=[1, 1, 1, 1], padding="SAME", name="conv_5_3")
    active_5_3 = tf.nn.relu(conv_5_3)
    pool5 = tf.nn.max_pool(active_5_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")

    pre_fully_connected = tf.contrib.layers.flatten(pool5)
    fully_connected_1 = tf.layers.dense(pre_fully_connected, 410, activation=tf.nn.relu, name="fc1")
    fully_connected_2 = tf.layers.dense(pre_fully_connected, 410, activation=tf.nn.relu, name="fc2")
    logits = tf.layers.dense(fully_connected_1, n_outputs, activation=tf.nn.relu, name="fc3")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="softmax"))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    n_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))


init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto(device_count={'GPU': 1})
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    init.run()
    batch_number = 1
    total_correct = 0
    for epoch in range(N_EPOCHS):
        csv_gen = csv_generator(DIRPATH, BATCH_SIZE)
        while True:
            try:
                X_batch, Y_batch = get_batch(csv_gen, label_to_class, class_eye)
            except StopIteration:
                break
            sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})
            total_correct += n_correct.eval({X: X_batch, Y: Y_batch})
            train_accuracy = total_correct / (batch_number * BATCH_SIZE)
            print("Epoch:", epoch + 1, "Batch Number:", batch_number, "Train accuracy:", train_accuracy)
            batch_number += 1
            if batch_number == 10:
                break

    print("Total accuracy: ", total_correct / (csv_len * N_EPOCHS))
    save_path = saver.save(sess, "./quick_draw_model")
