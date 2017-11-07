import numpy as np
import cv2


class SequenceImagesProvider:
    def __init__(self, sequence_path, input_dimensions=(320, 240)):
        """
        Class for handling video stream.
        :param sequence_path: path to video file
        :param input_dimensions: dimensions which are returned from stream
        """
        self.input_dimensions = input_dimensions
        self.cap = cv2.VideoCapture(sequence_path)
        while not self.cap.isOpened():
            self.cap = cv2.VideoCapture(sequence_path)
            print("Cannot open video stream.")
            cv2.waitKey(1000)

        print("Video loaded properly.")

        # utils
        self.counter = 0
        self.total_frames = self.cap.get(7)

    def get_ref_frame(self):
        self.cap.set(1, 0)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, self.input_dimensions)
        return frame

    def __call__(self):
        """
        Call returns next frame.
        :return: Next frame from sequence.
        """
        # check counter
        if self.counter >= self.total_frames:
            self.counter = 0

        # set current frame
        print(self.counter)
        self.cap.set(1, self.counter)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, self.input_dimensions)
        self.counter = self.counter + 1
        return frame
