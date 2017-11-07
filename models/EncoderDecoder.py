import tensorflow as tf
import numpy as np
import math


class Normal:
    def __init__(self, input_output_size=(320, 240, 3)):
        """
        Encoder-decoder net for unsupervised learning.
        :param input_output_size: size of input image
        """
        self.IMG_WIDTH = input_output_size[0]
        self.IMG_HEIGHT = input_output_size[1]
        self.IMG_CHANNELS = input_output_size[2]

        # Input and target placeholders
        self.inputs1 = tf.placeholder(tf.float32, (None, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), name="input1")
        self.inputs2 = tf.placeholder(tf.float32, (None, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), name="input2")
        # self.target = tf.placeholder(tf.float32, (None, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS), name="prediction")

        self.norm = 0
        self.model_initialized = False

    @staticmethod
    def encode(self, inputs):
        with tf.variable_scope("conv1"):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            # Now 320x240x16
            maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')

        with tf.variable_scope("conv2"):
            # Now 160x120x16
            conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            # Now 160x120x8
            maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

        with tf.variable_scope("conv3"):
            # Now 80x60x8
            conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            # Now 80x60x8
            encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same', name="encoded_filters")

        # Now 40x30x8
        return encoded

    @staticmethod
    def decode(self, inputs):
        with tf.variable_scope("upsample"):
            ### Decoder
            upsample1 = tf.image.resize_images(inputs, size=(60, 80), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with tf.variable_scope("conv4"):
            # Now 80x60x8
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            # Now 80x60x8
            upsample2 = tf.image.resize_images(conv4, size=(120, 160), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with tf.variable_scope("conv5"):
            # Now 160x120x8
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
            # Now 160x120x8
            upsample3 = tf.image.resize_images(conv5, size=(240, 320), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with tf.variable_scope("conv6"):
            # Now 320x240x8
            conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

        with tf.variable_scope("result_decoded"):
            # Now 320x240x16
            logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3, 3), padding='same', activation=None)

            # Now 320x240x3
            # Pass logits through tanh to get reconstructed image
            decoded = tf.nn.tanh(logits)

        return decoded

    def init_structure(self):

        # share weights and biases while encoding
        with tf.variable_scope("image_filters") as scope:
            encoded_1 = self.encode(self, inputs=self.inputs1)
            scope.reuse_variables()
            encoded_2 = self.encode(self, inputs=self.inputs2)

        # diff latent vectors
        diff = tf.subtract(encoded_1, encoded_2)
        self.norm = tf.norm(diff, ord='euclidean', name='norm')
        self.model_initialized = True

        dropout = tf.layers.dropout(diff, rate=0.5)

        # encode difference
        decoded = self.decode(self, inputs=dropout)

        return [decoded]

    def __call__(self, sess, feed, evaluation):
        return sess.run([evaluation], feed_dict={self.inputs1: feed[0], self.inputs2: feed[1]})

    def get_value_to_minimize(self):
        if self.model_initialized:
            return self.norm
        else:
            raise Exception("Model is not initialized. Nothing to optimize!")
