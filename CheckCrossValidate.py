"""
When run in main, module will test the cross validation set on several models on the specified data set
"""
from AuxiliaryCNN import csv_generator, text_to_labels, get_batch
import tensorflow as tf
import numpy as np
import sys


def main(file_to_test):
    """
    Driver

    :param file_to_test: (string)
    :return: VOID
    """
    DATA_DIR = "/data/scratch/epeake/Google-Doodles/"
    DIR_PATHS = ["./qd_model/lr-0.0003-mt-AlexDeep/",
                "./qd_model/lr-0.0003-mt-Alex/",
                "./qd_model/lr-0.0003-mt-AlexDeep2/"]

    savers = [tf.train.import_meta_graph(path + "cnnmodel-26031.meta", clear_devices=True) for path in DIR_PATHS]

    label_to_index = text_to_labels(DATA_DIR)
    class_eye = np.eye(len(label_to_index))
    n_outputs = len(label_to_index)
    BATCH_SIZE = 1
    HEIGHT = 256
    WIDTH = 256

    for i in range(len(savers)):
        tf.reset_default_graph()
        with tf.Session() as sess:
            savers[i].restore(sess, tf.train.latest_checkpoint(DIR_PATHS[i]))
            graph = tf.get_default_graph()

            X = tf.placeholder("float", [None, HEIGHT, WIDTH, 1], name="X")
            Y = tf.placeholder("float", [None, n_outputs], name="Y")

            conv_1_1 = tf.layers.conv2d(X, filters=96, kernel_size=11, strides=(4, 4),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="VALID",
                                        activation=tf.nn.relu, name="conv_1_1")
            pool1 = tf.nn.max_pool(conv_1_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

            conv_2_1 = tf.layers.conv2d(pool1, filters=256, kernel_size=5, strides=(1, 1),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="VALID",
                                        activation=tf.nn.relu, name="conv_2_1")
            pool2 = tf.nn.max_pool(conv_2_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

            conv_3_1 = tf.layers.conv2d(pool2, filters=348, kernel_size=3, strides=(1, 1),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="SAME",
                                        activation=tf.nn.relu, name="conv_3_1")
            conv_3_2 = tf.layers.conv2d(conv_3_1, filters=348, kernel_size=3, strides=(1, 1),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="SAME",
                                        activation=tf.nn.relu, name="conv_3_2")
            conv_3_3 = tf.layers.conv2d(conv_3_2, filters=256, kernel_size=3, strides=(1, 1),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="SAME",
                                        activation=tf.nn.relu, name="conv_3_3")
            pool3 = tf.nn.max_pool(conv_3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

            pre_fully_connected = tf.contrib.layers.flatten(pool3, scope="flattened")

            fully_connected_1 = tf.layers.dense(pre_fully_connected, 4010,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                activation=tf.nn.relu, name="fc1")

            fully_connected_2 = tf.layers.dense(fully_connected_1, 4010,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                activation=tf.nn.relu, name="fc2")

            logits = tf.layers.dense(fully_connected_2, n_outputs, activation=tf.nn.relu, name="logits")

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy_op")

            csv_gen = csv_generator(DATA_DIR, BATCH_SIZE, file_name=file_to_test, shuffle=False)
            test_accuracy = 0
            batch_number = 1
            while True:
                try:
                    X_batch, Y_batch = get_batch(csv_gen, label_to_index, class_eye)
                    test_accuracy += sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
                    if batch_number % 1000 == 0:
                        print(test_accuracy)

                except StopIteration:
                    test_accuracy = test_accuracy / batch_number
                    break

                batch_number += 1

            print(DIR_PATHS[i], file_to_test, "Accuracy =", test_accuracy)


if __name__ == '__main__':
    main(sys.argv[1])
