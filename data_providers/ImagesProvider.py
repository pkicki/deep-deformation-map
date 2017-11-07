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
        self.counter = 1
        self.total_frames = self.cap.get(7)
        self.first_pass = True

    def get_ref_frame(self):
        self.cap.set(1, 0)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, self.input_dimensions)
        return frame

    def is_first_pass(self):
        return self.first_pass

    def get_current_frame_no(self):
        return self.counter

    def __call__(self):
        """
        Call returns next frame.
        :return: Next frame from sequence.
        """
        # check counter
        if self.counter >= int(self.total_frames):
            self.counter = 1
            self.first_pass = False

        # set current frame
        self.cap.set(1, self.counter)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, self.input_dimensions)

        self.counter = self.counter + 1
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()