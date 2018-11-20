from AuxiliaryCNN import csvManager, text_to_labels, get_batch
import numpy as np
import tensorflow as tf

# for debugging
tf.set_random_seed(1)
np.random.seed(1)

FILEPATH = "/Users/epeake/Desktop/Google-Doodles/"
BATCH_SIZE = 100
height = 256
width = 256
csvM = csvManager(FILEPATH)
csvM.open_files()
label_to_class = text_to_labels(csvM)
class_eye = np.eye(len(label_to_class))

# conv1 params
conv1_n_fills = 10
conv1_kernel = 3
conv1_stride = 1
conv1_pad = "same"

# conv2 params
conv2_n_fills = 20
conv2_kernel = 5
conv2_stride = 2
conv2_pad = "valid"

# logits
n_outputs = len(label_to_class)

# optimization
learning_rate = 0.009


X = tf.placeholder("float", [None, height, width, 1])   # [None, height, width, channels]
Y = tf.placeholder("float", [None, n_outputs])       # [None, classes]

conv1 = tf.layers.conv2d(X, filters=conv1_n_fills, kernel_size=conv1_kernel,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")

pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=1, padding='valid', name="pool1")

conv2 = tf.layers.conv2d(pool1, filters=conv2_n_fills, kernel_size=conv2_kernel,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

pool2 = tf.layers.max_pooling2d(conv2, pool_size=(5, 5), strides=2, padding='valid', name="pool2")

pre_fully_connected = tf.contrib.layers.flatten(pool2)

fully_connected_1 = tf.layers.dense(pre_fully_connected, 64, activation=tf.nn.relu, name="fc1")

logits = tf.layers.dense(fully_connected_1, n_outputs, activation=tf.nn.relu, name="fc2")

# TODO: Fix loss function and check architecture
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="softmax"))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct = tf.nn.in_top_k(logits, Y, 1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


n_epochs = 10
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        csvM = csvManager(FILEPATH)
        csvM.open_files()
        X_len = 1
        while X_len:
            # this cycle is for dividing step by step the heavy work of each neuron
            X_batch, Y_batch = get_batch(csvM, label_to_class, class_eye, BATCH_SIZE)
            sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})
            X_len = len(X_batch)
            n_correct = correct.eval(feed_dict={Y: Y_batch})
            acc_train = n_correct / len(Y_batch)
            print("Epoch:", epoch + 1, "Train accuracy:", acc_train)
        csvM.close_files()

    save_path = saver.save(sess, "./my_fashion_model")


# TODO: Make run in command line
# if __name__ == "__main__":
#     filepath = sys.argv[1]
