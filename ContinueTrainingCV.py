"""
When run in main, module will continue training a saved model with the cross validation set
"""
from AuxiliaryCNN import csv_generator, text_to_labels, get_batch
import numpy as np
import tensorflow as tf


def main():
    """
    Driver

    :return: VOID
    """
    DATA_DIR = "/data/scratch/epeake/Google-Doodles/"
    MODEL_PATH = "./qd_model/lr-0.0003-mt-AlexDeep/"
    BATCH_SIZE = 40
    N_EPOCHS = 2
    LEARNING_RATE = 0.0003

    label_to_index = text_to_labels(DATA_DIR)
    class_eye = np.eye(len(label_to_index))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH + "cnnmodel-26031.meta", clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        graph = tf.get_default_graph()

        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        softmax = graph.get_tensor_by_name("softmax:0")
        loss = tf.reduce_mean(softmax)
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
        accuracy = graph.get_tensor_by_name("accuracy:0")

        batch_number = 1
        for epoch in range(N_EPOCHS):
            csv_gen = csv_generator(DATA_DIR, BATCH_SIZE, file_name="cross_validate.csv")
            while True:
                try:
                    X_batch, Y_batch = get_batch(csv_gen, label_to_index, class_eye)
                except StopIteration:
                    break

                sess.run(train_step, feed_dict={X: X_batch, Y: Y_batch})
                if batch_number % 100 == 0:
                    batch_accuracy = sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
                    print("Epoch:", epoch + 1, "Total Batch Number:", batch_number, "Train accuracy:", batch_accuracy)
                batch_number += 1

        saver.save(sess, MODEL_PATH + "final-model/cnnmodel")


if __name__ == '__main__':
    main()
