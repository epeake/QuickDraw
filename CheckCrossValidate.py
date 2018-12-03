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
    BATCH_SIZE = 1

    for i in range(len(savers)):
        with tf.Session() as sess:
            savers[i].restore(sess, tf.train.latest_checkpoint(DIR_PATHS[i]))
            graph = tf.get_default_graph()

            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")
            accuracy = graph.get_tensor_by_name("accuracy:0")
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
