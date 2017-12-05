import tensorflow as tf
import cv2
import numpy as np
import matplotlib.colors as cl
import math
from math import sin, cos
from scipy import ndimage
import csv
from matplotlib import pyplot as plt


class MBTester:
    def __init__(self,
                 data_provider,
                 net_name,
                 flownet_provider,
                 roi_width=150,
                 roi_height=200,
                 output_path="",
                 test_method="human",
                 display_threshold=255):

        self.net_name = net_name
        self.data_provider = data_provider
        self.output_path = output_path
        self.display_threshold = display_threshold
        self.flownet_provider = flownet_provider
        self.roi_width = roi_width
        self.roi_height = roi_height

        # self.mask = [(227, 100), (350, 106), (515, 122), (510, 250), (506, 367), (360, 350), (208, 343), (210, 180)]
        # self.mask = [(56, 25), (81, 26), (125, 30), (127, 88), (126, 91), (90, 88), (52, 85), (53, 45)]
        self.mask = []  # user specified mask
        self.mask_float = [] # continous mask
        self.reference_frame = np.zeros(0)
        self.scale = 1

        # available testing modes
        if test_method in ["human", "file", "flownet", "flow_deform"]:
            self.test_method = test_method
        else:
            raise SyntaxError("No such test method")

    def click_and_crop(self, event, x, y, flags, param):
        """
        If the left mouse button was clicked, record the starting
        (x, y) coordinates and add them to mask points
        :param event:
        :param x: coordinate x of mask point
        :param y: coordinate y of mask point
        :param flags:
        :param param:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            point_cv = (x, y)
            point_py = (y, x)
            self.mask.append(point_py)
            self.mask_float.append(point_py)
            cv2.circle(self.reference_frame, point_cv, 5, (0, 0, 255), -1)

    def init_mask(self, image):
        """
        Initialize mask to track.
        :param image: reference image
        :return:
        """
        self.reference_frame = image.copy()
        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.reference_frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.reference_frame = clone.copy()

            if key == ord("s"):
                self.save_mask()

            if key == ord("l"):
                self.load_mask()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        cv2.destroyAllWindows()

    def save_mask(self):
        with open("last_mask.tsv", "wb") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(self.mask)
            print("Mask saved")

    def load_mask(self):
        with open("last_mask.tsv", "rb") as f:
            reader = csv.reader(f, delimiter="\t")
            self.mask = [(int(point[0]), int(point[1])) for point in reader]
            self.mask_float = [x[:] for x in self.mask]
            print("Mask loaded")


    @staticmethod
    def compute_vectors_length(self, mat):
        """
        Compute lengths of flow vectors and add third column to flow matrix.
        :param self:
        :param mat:
        :return:
        """
        vector_length_mat = np.zeros(shape=(mat.shape[0], mat.shape[1]))

        # create matrix of flow vectors lengths
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                element = mat[i][j]
                length = math.sqrt(math.pow(element[0], 2) + math.pow(element[1], 2))
                vector_length_mat[i][j] = length

        return vector_length_mat

    @staticmethod
    def compute_roi(self, cx, cy):
        """
        Compute rectangular roi for specified center of mass.
        :param self:
        :param cx: x coordinate of center of mass.
        :param cy: y coordinate of center of mass.
        :return:
        """
        offset = 100

        # boundaries
        left = 0 if (cx - self.roi_width/2) < 0 else cx - self.roi_width/2
        right = 0 if (cx + self.roi_width/2) < 0 else cx + self.roi_width/2
        up = 0 if (cy - self.roi_height/2) < 0 else cy - self.roi_height/2
        down = 0 if (cy + self.roi_height/2) < 0 else cy + self.roi_height/2

        return int(left), int(right), int(up), int(down)

    @staticmethod
    def flow2rgb(self, flow):
        """
        Take flow matrix and encode it as RGB. Just for simple visualization.
        :param self:
        :param flow: flow matrix. Output from Caffe model.
        :return: RGB matrix representing optical flow
        """
        # split flow into channels
        b_channel, g_channel = cv2.split(flow)

        # scale channels
        b_channel = b_channel
        g_channel = g_channel
        r_channel = np.ones((flow.shape[0], flow.shape[1]), dtype=b_channel.dtype)

        # merge them to BGR
        return_img = cv2.merge((b_channel, g_channel, r_channel))
        return return_img

    @staticmethod
    def decode_flownet_output(self, flow):
        """
        Decode flownet2.0 output to angle and distance of pixel displacement.
        :param image:
        :param self:
        :param flow: output from flownet2.0
        :return:
        """

        UNKNOWN_FLOW_THRESH = 1e7

        u = flow[:, :, 0]
        v = flow[:, :, 1]

        # remove errors in flow vectors
        idx_unknown = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idx_unknown] = 0
        v[idx_unknown] = 0

        # normalize to (-1, 1)
        # rad = np.sqrt(u ** 2 + v ** 2)
        # maxrad = max(-1, np.max(rad))
        #
        # u = u / (maxrad + np.finfo(float).eps)
        # v = v / (maxrad + np.finfo(float).eps)

        radians = np.arctan2(-u, -v)            # obtain radians
        magnitudes = np.sqrt(u ** 2 + v ** 2)   # obtain displacement in pixels

        # magnitudes thresholding
        max_magnitude = np.max(magnitudes)
        magnitudes = np.where(magnitudes > max_magnitude/3, magnitudes, 0)
        magnitudes = np.where(magnitudes < 30, magnitudes, 0)

        H, W = magnitudes.shape

        magnitudes_shifted = np.zeros((H, W))
        radians_shifted = np.zeros((H, W))

        for i in range(H):
            for k in range(W):
                new_x = k + int(round(magnitudes[i][k] * sin(radians[i][k])))
                new_y = i + int(round(magnitudes[i][k] * cos(radians[i][k])))
                if new_x < W and new_x >= 0 and new_y < H and new_y >= 0:
                    magnitudes_shifted[new_y][new_x] = magnitudes[i][k]
                    radians_shifted[new_y][new_x] = radians[i][k]

        return radians_shifted, magnitudes_shifted

    def update_mask(self, radians, magnitudes):
        """
        Update mask to track using flownet2.0.
        :param radians:
        :param magnitudes: distance of pixel displacement towards angle
        :return:
        """
        assert(radians.shape == magnitudes.shape)

        print("\nRegion to be tracked: ")
        for i in range(len(self.mask)):

            height = self.reference_frame.shape[0]
            width = self.reference_frame.shape[1]
            print("height %d, width %d" % (height, width))

            # get mask indexes
            idy = self.mask[i][0]
            idx = self.mask[i][1]
            idy_float = self.mask_float[i][0]
            idx_float = self.mask_float[i][1]

            # check indexes
            if idx >= width:
                idx = width - 1
            if idy >= height:
                idy = height - 1

            # get parameters of displacement
            radian = radians[idy, idx]
            displ = magnitudes[idy, idx]
            print("Point (%d, %d) moved by: %f px at %f radians." % (idx, idy, displ, radian))

            # compute new positions taking into consideration that coordinates are rotated by +pi/2
            # TODO consider if it is enough to get good transform and fix if not
            bias = 0.04
            new_x = idx_float + displ * math.sin(radian + bias)
            new_y = idy_float + displ * math.cos(radian + bias)


            if new_x >= width:
                new_x = width - 1
            if new_y >= height:
                new_y = height - 1

            # update mask
            self.mask[i] = (int(round(new_y)), int(round(new_x)))
            self.mask_float[i] = (new_y, new_x)

    def get_mask(self, flow, image):
        """
        Compute new shape of mask using opticalflow.
        :param flow:
        :param image:
        :return:
        """
        # compute points displacement
        r, m = self.decode_flownet_output(self, flow)
        self.update_mask(r, m)

        # fill the ROI so it doesn't get wiped out when the mask is applied
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask_T = [ (y, x) for x, y in self.mask ]
        roi_corners = np.array([mask_T], dtype=np.int32)
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)

        return mask

    def test(self):
        """
        Wrapper for user-specified testing methods.
        :return:
        """
        if self.test_method == "flownet":

            # setup flow net2.0 and test it -- CAFFE
            firs_iteration_pass = False
            previous_frame = self.data_provider()
            i = 1
            r = 7
            while True:

                if cv2.waitKey(1) == ord('s'):
                    while True:
                        if cv2.waitKey(1) == ord('s'):
                            break
                if cv2.waitKey(1) == ord('q'):
                    break

                frame = self.data_provider()
                if firs_iteration_pass and i%r == 0:
                    flow = self.flownet_provider(img0=previous_frame, img1=frame)  # 240x320x2 -> HxWx2
                    a, m = self.decode_flownet_output(self, flow)

                    plt.figure()
                    plt.subplot(2, 2, 1)
                    plt.imshow(previous_frame)
                    plt.subplot(2, 2, 2)
                    plt.imshow(frame)
                    plt.subplot(2, 2, 3)
                    plt.imshow(a)
                    plt.subplot(2, 2, 4)
                    plt.imshow(m)
                    plt.draw()
                    plt.waitforbuttonpress()
                    #plt.pause(0.1)
                    plt.close()

                    previous_frame = frame

                    # mass center of returned flow -- object tracking
                    # vec_lengths = self.compute_vectors_length(self, return_img)
                    # center_x, center_y = np.floor(ndimage.measurements.center_of_mass(vec_lengths))
                    # x_from, x_to, y_from, y_to = self.compute_roi(self, center_x, center_y)
                    # vis = frame[x_from:x_to, y_from:y_to]

                cv2.imshow("output", frame)
                firs_iteration_pass = True
                i += 1

        else:

            # setup session -- TENSORFLOW
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            with tf.Session(config=config) as sess:
                saver = tf.train.import_meta_graph(self.net_name)
                saver.restore(sess, tf.train.latest_checkpoint('./'))

                input1 = sess.graph.get_tensor_by_name("input1:0")
                input2 = sess.graph.get_tensor_by_name("input2:0")
                output1 = sess.graph.get_tensor_by_name("result_decoded/Tanh:0")

                # setup needed variables
                firs_iteration_pass = False
                first_out = False
                previous_frame = self.data_provider()#np.zeros(0)
                ref_img = self.data_provider.get_ref_frame()

                # initialize mask to track using optical flow
                self.init_mask(ref_img)
                assert(self.mask != [])

                i = 1
                r = 3

                # run tracking using specified test method
                while True:

                    if not self.data_provider.is_first_pass():
                        break

                    if cv2.waitKey(1) == ord('s'):
                        while True:
                            if cv2.waitKey(1) == ord('s'):
                                break
                    if cv2.waitKey(1) == ord('q'):
                        break

                    # capture frame
                    frame = self.data_provider()
                    frame_no = self.data_provider.get_current_frame_no()

                    # just tensorflow output - photo one by one
                    if self.test_method == "human":

                        if cv2.waitKey(1) == ord('s'):
                            while True:
                                if cv2.waitKey(1) == ord('s'):
                                    break
                        if cv2.waitKey(1) == ord('q'):
                            break

                        return_img = sess.run(output1[0], feed_dict={input1: [ref_img],
                                                                     input2: [frame]})
                        cv2.imshow("output", return_img)

                    # just tensorflow output - sequence -> save to files
                    if self.test_method == "file":

                        return_img = sess.run(output1[0], feed_dict={input1: [ref_img],
                                                                     input2: [frame]}) * self.display_threshold

                        name = "{}frame_{:03d}.jpg".format(self.output_path, frame_no)
                        cv2.imwrite(name, return_img)

                    # combine roi tracking with tensorflow output - sequence
                    if self.test_method == "flow_deform":

                        frame = self.data_provider()

                        if firs_iteration_pass and i % r == 0:

                            # get roi using flownet
                            return_flow = self.flownet_provider(img0=frame,
                                                                img1=previous_frame)
                            mask = self.get_mask(return_flow, frame)

                            # get deformations using my net
                            return_img = sess.run(output1[0], feed_dict={input1: [ref_img],
                                                                         input2: [frame]})
                            return_img = np.array(return_img) * self.display_threshold   # generic scaling

                            # track object using flow net - apply proper mask
                            # scale images to float 0 - 1 for easier math
                            frame_masked = np.array(frame, dtype=np.float)
                            frame_masked /= 255.0
                            frame_masked *= .25  # transparency

                            mask = np.array(mask, dtype=np.float)
                            mask /= 255.0

                            return_img = np.array(return_img, dtype=np.float)
                            return_img /= 255.0

                            # apply mask
                            out = return_img * mask + frame_masked * (1.0 - mask)
                            first_out = True

                            a, m = self.decode_flownet_output(self, return_flow)

                            plt.imshow(m)
                            plt.draw()
                            plt.pause(0.1)

                        if first_out:
                            cv2.imshow("output", out)

                        if i % r == 0:
                            previous_frame = frame
                        firs_iteration_pass = True
                        i += 1

                print("Tests finished!")
