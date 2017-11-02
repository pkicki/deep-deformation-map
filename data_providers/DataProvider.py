import os
import numpy as np
from scipy.ndimage import imread


class Producer:
    def __init__(self, path, batch_size=32, input_image_size=(320, 240)):
        self.path = path
        self.batch_size = batch_size
        self.input_image_size = input_image_size

        self.counter = 0
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []
        self.is_init = False

    def load(self):
        data, labels = list(), list()

        # Read data set file
        if os.path.exists(self.path):
            file = open(self.path, 'r').read().splitlines()
            for f in file:
                image1 = imread(f.split(' ')[0]).resize(self.input_image_size)
                image2 = imread(f.split(' ')[1]).resize(self.input_image_size)
                data.append([image1, image2])
                labels.append([float(f.split(' ')[2]), float(f.split(' ')[3])])

            # must be equal
            assert (len(labels) == len(data))

            split = int(len(data) * 0.75)
            data = np.array(data)

            self.train_data = data[:split]
            self.test_data = data[split:]
            self.train_labels = labels[:split]
            self.test_labels = labels[split:]

            self.is_init = True

    def __call__(self):
        if self.is_init:

            if self.counter + self.batch_size > len(self.train_data):
                self.counter = 0

            obj = [
                self.train_data[self.counter:self.counter+self.batch_size],
                self.test_data[self.counter:self.counter+self.batch_size],
                self.train_labels[self.counter:self.counter+self.batch_size],
                self.test_labels[self.counter:self.counter+self.batch_size]
            ]
            self.counter = self.counter + self.batch_size
            return obj

        raise RuntimeError("Initialize object first using load() method!")
