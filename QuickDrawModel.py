from AuxiliaryCNN import csvManager, text_to_labels, get_batch
import numpy as np
import tensorflow as tf

FILEPATH = "/data/scratch/epeake/Google-Doodles/"
BATCH_SIZE = 100
height = 256
width = 256
csvM = csvManager(FILEPATH)
csvM.open_files()
label_to_class = text_to_labels(csvM)
class_eye = np.eye(len(label_to_class))
n_outputs = len(label_to_class)
learning_rate = 0.009

with tf.device("/gpu:0"):
    X = tf.placeholder("float", [None, height, width, 1])   # [None, height, width, channels]
    Y = tf.placeholder("float", [None, n_outputs])       # [None, classes]

    W1 = tf.get_variable("W1", [3, 3, 1, 10], initializer=tf.contrib.layers.xavier_initializer())
    conv1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
    active1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(active1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID', name="pool1")

    W2 = tf.get_variable("W2", [5, 5, 10, 20], initializer=tf.contrib.layers.xavier_initializer())
    conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 2, 2, 1], padding="SAME", name="conv2")
    active2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(active2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")

    pre_fully_connected = tf.contrib.layers.flatten(pool2)
    fully_connected_1 = tf.layers.dense(pre_fully_connected, 64, activation=tf.nn.relu, name="fc1")
    logits = tf.layers.dense(fully_connected_1, n_outputs, activation=tf.nn.relu, name="fc2")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="softmax"))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    n_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.allow_soft_placement = True
saver = tf.train.Saver()

n_epochs = 10
with tf.Session(config) as sess:
    init.run()
    for epoch in range(n_epochs):
        total_correct = 0
        csvM = csvManager(FILEPATH)
        csvM.open_files()
        X_len = 1
        batch_number = 1
        while X_len:
            # this cycle is for dividing step by step the heavy work of each neuron
            X_batch, Y_batch = get_batch(csvM, label_to_class, class_eye, BATCH_SIZE)
            sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})
            total_correct += n_correct.eval({X: X_batch, Y: Y_batch})
            train_accuracy = total_correct / (batch_number * BATCH_SIZE)
            print("Epoch:", epoch + 1, "Batch Number:", batch_number, "Train accuracy:", train_accuracy)
            X_len = len(X_batch)
            batch_number += 1
        csvM.close_files()

    save_path = saver.save(sess, "./quick_draw_model")
