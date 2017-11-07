import os
import numpy as np
import cv2


class Producer:
    def __init__(self, path, batch_size=32, input_image_size=(320, 240)):
        self.path = path
        self.batch_size = batch_size
        self.input_image_size = input_image_size

        self.counter = 0
        self.train_data = np.zeros(0)
        self.test_data = np.zeros(0)
        self.train_labels = np.zeros(0)
        self.test_labels = np.zeros(0)
        self.is_init = False

    def load(self):
        data, labels = list(), list()

        # Read data set file
        if os.path.exists(self.path):
            file = open(self.path, 'r').read().splitlines()
            for f in file:
                data.append([f.split(' ')[0], f.split(' ')[1]])
                labels.append([f.split(' ')[2]])

            # must be equal
            assert (len(labels) == len(data))

            split = int(len(data) * 0.75)
            data = np.array(data)

            self.train_data = data[:split]
            self.test_data = data[split:]
            self.train_labels = labels[:split]
            self.test_labels = labels[split:]

            self.is_init = True

    def get_images(self, names, out_images, out_labels):
        input_11 = cv2.imread(names[0], cv2.IMREAD_COLOR)
        input_12 = cv2.imread(names[1], cv2.IMREAD_COLOR)
        input_11 = cv2.resize(input_11, self.input_image_size)
        input_12 = cv2.resize(input_12, self.input_image_size)
        input_11 = input_11 * 1.0 / 127.5 - 1.0
        input_12 = input_12 * 1.0 / 127.5 - 1.0
        out_images.append([input_11, input_12])

        # for supervised
        output_1 = cv2.imread(names[1], cv2.IMREAD_COLOR)
        output_1 = cv2.resize(output_1, self.input_image_size)
        output_1 = output_1 * 1.0 / 127.5 - 1.0
        out_labels.append(output_1)

        return [out_images, out_labels]

    def __call__(self):
        if self.is_init:

            if self.counter + self.batch_size > len(self.train_data):
                self.counter = 0

            train = list()
            test = list()
            train_labels = list()
            test_labels = list()

            for i in range(self.counter, self.counter+self.batch_size):
                out_train = self.get_images(names=self.train_data[i], out_images=train, out_labels=train_labels)
                # out_test = self.get_images(names=self.test_data[i], out_images=test, out_labels=test_labels)

            train = np.array(out_train[0])
            # test = np.array(out_test[0])

            obj = [
                train,
                test,
                train_labels,
                test_labels
            ]
            self.counter = self.counter + self.batch_size
            return obj

        raise RuntimeError("Initialize object first using load() method!")
