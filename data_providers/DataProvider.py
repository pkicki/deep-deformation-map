import os
import numpy as np
from scipy.ndimage import imread


class Producer:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.counter = 0
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []
        self.is_init = False

    def load(self):
        data, labels = list(), list()

        # Read dataset file
        if os.path.exists(self.path):
            file = open(self.path, 'r').read().splitlines()
            for f in file:
                data.append([imread(f.split(' ')[0]), imread(f.split(' ')[1])])
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
        print(self.is_init)
        if self.is_init:
            obj = [
                self.train_data[self.counter:self.counter+self.batch_size],
                self.test_data[self.counter:self.counter+self.batch_size],
                self.train_labels[self.counter:self.counter+self.batch_size],
                self.test_labels[self.counter:self.counter+self.batch_size]
            ]
            self.counter = self.counter + self.batch_size
            return obj

        raise RuntimeError("Initialize object first using load() method!")


    # def __init__(self, img_path, window_size=(64, 64), step_size=1):
    #     self.image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     self.window_size = window_size
    #     self.step_size = step_size
    #
    # def get_image(self):
    #     return self.image
    #
    # def sliding_window(self):
    #     # slide a window across the image
    #     for x in range(self.step_size, self.image.shape[1] - self.step_size, self.step_size):
    #         for y in range(self.step_size, self.image.shape[0] - self.step_size, self.step_size):
    #             # yield the current window
    #             yield (x + int(self.window_size[0]/2), y + int(self.window_size[0]/2), self.image[y:y + self.window_size[1], x:x + self.window_size[0]])
    #
    # def __call__(self):
    #     # get the cropped list of patches and its center points
    #     return self.sliding_window()
