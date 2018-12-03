"""
Module contains the auxiliary functions necessary to train a CNN for Google's Quick Draw competition.

Functions are optimized to handle massive datasets without using too much RAM.
"""
import os
import numpy as np
from subprocess import call, check_output


def process_lines(raw_lines):
    """
    Converts a list of lines into a list of usable doodle entries to be drawn

    :param raw_lines: (list of strings
    :return: (list of dictionaries) our entries
    """
    lines = []
    for line in raw_lines:
        split_line = line.split('"')  # split line so we can extract points
        pre_points = eval(split_line[1])
        points = []
        for segment in pre_points:
            points.append([(x, y) for x, y in zip(segment[0], segment[1])])
        part_of_line = split_line[2].split(",")
        recognized = eval(part_of_line[2])
        label = part_of_line[-1].strip()
        lines.append({"points": points, "recognized": recognized, "label": label})

    return lines


def csv_generator(dir_path, batch_size, file_name="train.csv", shuffle=True):
    """
    Shuffles a csv file, then yields batches of its entries

    :param dir_path: (string)
    :param batch_size: (int)
    :param shuffle: (bool) should we shuffle our csv before taking batches?
    :param file_name: (string) file to be generated from
    :yield: (list of dictionaries) our batch's entries
    """
    if shuffle:
        # shuffle all entries (this can take a while if large)
        print("Shuffling entries")
        call("sort -R -o " + dir_path + file_name + " " + dir_path + file_name, shell=True)
        print("Shuffling complete")
    csv_len = int(check_output('wc -l ' + dir_path + file_name + ' | grep -o "[0-9]\+"', shell=True))
    with open(dir_path + file_name) as file:
        for i in range(csv_len // batch_size):
            lines = []
            for _ in range(batch_size):
                lines.append(file.readline())
            yield process_lines(lines)

        lines = []
        for _ in range(csv_len % batch_size):
            lines.append(file.readline())
        yield process_lines(lines)


def get_pixels(points):
    """
    Calculates all the pixels needed to recreate an image based on a set of points and their connections

    :param points: (list of lists of tuples) each nested list represents a stroke and each tuple a point
    :return: (list of tuples) all pixels of the 256x256 image
    """
    pixels = []
    for group in points:
        for i in range(len(group)-1):

            # to avoid divide by 0 errors:
            if group[i][0] != group[i+1][0]:
                slope = (group[i][1] - group[i+1][1]) / (group[i][0] - group[i+1][0])
                b = group[i][1] - (slope * group[i][0])
                previous = min(group[i], group[i+1], key=lambda point: point[0])[1]   # y val of least x

                # look at corresponding y vals one x step at a time
                for x in range(min(group[i][0], group[i+1][0]), max(group[i][0], group[i+1][0]) + 1):
                    y = slope * x + b
                    rounded_y = int(y)

                    # draw down or up to connect the dots
                    for j in range(min(rounded_y, previous) + 1, max(rounded_y, previous)):
                        pixels.append((x, j))

                    pixels.append((x, rounded_y))
                    previous = rounded_y

            #  in this case, draw a line
            else:
                for y in range(min(group[i][1], group[i + 1][1]), max(group[i][1], group[i + 1][1]) + 1):
                    pixels.append((group[i][0], y))

    return pixels


def draw_picture(pixels):
    """
    Given a set of pixels for a 256x256 image, plots all pixels in a np.array

    :param pixels: (list of tuples)
    :return: (np.array) our picture
    """
    picture = np.zeros((256, 256))
    for pixel in pixels:
        picture[pixel] = 1

    return picture


def text_to_labels(dir_path):
    """
    Creates a dictionary with an index for each of the unique labels

    :param dir_path: (string)
    :return: (dictionary)
    """
    labels = [filename.replace(".csv", "") for filename in os.listdir(dir_path)
              if filename != "cross_validate.csv" and filename != "train.csv"
              and filename != "train_all.csv" and filename != "test.csv"
              and filename.find(".csv") != -1]
    label_to_index = {unique_label: i for i, unique_label in enumerate(labels)}
    return label_to_index


def class_to_one_hot(file_batch, label_to_index, class_eye):
    """
    Creates our Y matrix with one-hot encoding

    :param file_batch: (list of dictionaries) batch of pictures
    :param label_to_index: (dictionary) rosetta stone, labels to number
    :param class_eye: (np.array) identity matrix with length of num labels
    :return: (np.array) Y matrix with one-hot encoding
    """
    labels = np.array([label_to_index[file["label"]] for file in file_batch])
    return class_eye[labels]


def get_batch(csv_generator, label_to_index, class_eye):
    """
    Gets X and Y matrices of a specified batch size

    :param csv_generator: (generator) generator created by csv_generator function
    :param label_to_index: (dictionary) rosetta stone, labels to number
    :param class_eye: (np.array) identity matrix with length of num labels
    :return: (tuple of np.arrays)
    """
    X = []
    file_batch = next(csv_generator)
    Y = class_to_one_hot(file_batch, label_to_index, class_eye)
    for file in file_batch:
        pixels = get_pixels(file["points"])
        X.append(draw_picture(pixels))

    return np.expand_dims(X, axis=3), Y   # add chanel dim
