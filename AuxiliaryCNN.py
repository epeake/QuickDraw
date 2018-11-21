"""
Module contains the auxiliary functions necessary to train a CNN for Google's Quick Draw competition.

Functions are optimized to handle massive datasets without using too much RAM.
"""

import numpy as np
import os
import random


class csvManager:
    """
    Manager for csv files.  Can randomly read batches of entries from multiple csv files so we don't have to store
    entire csv files in memory
    """
    def __init__(self, filepath):
        """
        Constructor for csvManager.  Stores file names to be open in given filepath

        :param filepath: (string) Directory must contain exclusively csv files to be loaded
        """
        self.filepath = filepath
        self.files = os.listdir(filepath)
        self.files_opened = None

    def open_files(self):
        """
        Opens each of the files, stores them in a list, and removes the first line of colnames
        """
        if self.filepath[-1] != "/":   # so we can add the filename to the filepath
            self.filepath += "/"

        self.files_opened = [open(self.filepath + f) for f in self.files]
        for f in self.files_opened:
            next(f)  # skip colnames

    def close_files(self):
        """
        Closes all files.
        """
        if self.files_opened:
            for f in self.files_opened:
                f.close()

    def _try_line(self):
        """
        Returns a line if available.  Closes exhausted files along the way.

        :return: (string) line or None
        """
        if not self.files_opened:
            print("files exhausted")

        else:
            index = random.randint(0, len(self.files_opened) - 1)
            try:
                return next(self.files_opened[index])

            except StopIteration:  # file exhausted, try another
                self.files_opened[index].close()
                self.files_opened.pop(index)
                return self._try_line()

    def read_lines(self, n):
        """
        Read in n files from all stored, opened files.

        :param n: (int) number of files
        :return: (list of dictionaries) our entries
        """
        if not self.files_opened:
            print("files must first be opened")

        else:
            raw_lines = []
            for _ in range(n):
                line = self._try_line()   # need to see if has value or is none (files exhausted)
                if line:
                    raw_lines.append(line)
                else:
                    break

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


def text_to_labels(csvM):
    """
    Takes a csvManager and creates a dictionary with an index for each of the unique labels

    :param csvM: (csvManager)
    :return: (dictionary)
    """
    labels = [label.replace(".csv", "") for label in csvM.files]
    label_to_class = {unique_label: i for i, unique_label in enumerate(labels)}
    return label_to_class


def class_to_one_hot(file_batch, label_to_class, class_eye):
    """
    Creates our Y matrix with one-hot encoding

    :param file_batch: (list of dictionaries) batch of pictures
    :param label_to_class: (dictionary) rosetta stone, labels to number
    :param class_eye: (np.array) identity matrix with length of num labels
    :return: (np.array) Y matrix with one-hot encoding
    """
    labels = np.array([label_to_class[file["label"]] for file in file_batch])
    return class_eye[labels]


def get_batch(csvM, label_to_class, class_eye, batch_size):
    """
    Gets X and Y matrices of a specified batch size

    :param csvM: (csvManager)
    :param label_to_class: (dictionary) rosetta stone, labels to number
    :param class_eye: (np.array) identity matrix with length of num labels
    :param batch_size: (int)
    :return: (tuple of np.arrays)
    """
    X = []
    file_batch = csvM.read_lines(batch_size)
    Y = class_to_one_hot(file_batch, label_to_class, class_eye)
    for file in file_batch:
        pixels = get_pixels(file["points"])
        X.append(draw_picture(pixels))

    return np.expand_dims(X, axis=3), Y   # add chanel dim


def var_to_cpu(op, cpu, gpu):
    """
    Places an operation on the gpu unless it is a variable

    :param op: (tf op)
    :return: cpu or gpu device string
    """
    if op.type == "Variable":
        return "/cpu:0"

    else:
        return "/gpu:0"
