import tensorflow as tf
import cv2
import numpy as np
import matplotlib.colors as cl


class MBTester:
    def __init__(self, data_provider, net_name, flownet_provider, output_path="", test_method="human", display_threshold=255):
        self.net_name = net_name
        self.data_provider = data_provider
        self.output_path = output_path
        self.display_threshold = display_threshold
        self.flownet_provider = flownet_provider

        if test_method in ["human", "file", "flownet"]:
            self.test_method = test_method
        else:
            raise SyntaxError("No such test method")

    @staticmethod
    def flow2rgb_optflowtoolkit(self, flow):

        print(flow.shape)
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        #valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)

        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)

        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow

        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]

        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1

        # convert to rgb
        img = cl.hsv_to_rgb(img)

        return img

    @staticmethod
    def flowflow2rgb_handcraft(self, flow):

        # split flow into channels
        b_channel, g_channel = cv2.split(flow)

        # scale channels
        b_channel = b_channel
        g_channel = g_channel
        r_channel = np.ones((flow.shape[0], flow.shape[1]), dtype=b_channel.dtype)

        # merge them to BGR
        return_img = cv2.merge((b_channel, g_channel, r_channel))
        return return_img

    def test(self):

        if self.test_method == "flownet":

            # setup reference image
            cnt = 0
            previous_frame = np.zeros(0)
            while True:

                if cv2.waitKey(1) == ord('s'):
                    while True:
                        if cv2.waitKey(1) == ord('s'):
                            break
                if cv2.waitKey(1) == ord('q'):
                    break

                frame = self.data_provider()
                if cnt > 0:
                    # TODO: convert return_img (320x240x2 -> 320x240x3) and normalize
                    return_img = self.flownet_provider(img0=previous_frame, img1=frame)
                    return_img = self.flowflow2rgb_handcraft(self, return_img)

                    # output = cv2.normalize(return_img, return_img, 0, 255, cv2.NORM_MINMAX)
                    cv2.imshow("b_channel", return_img)
                    cv2.waitKey(10000)

                previous_frame = frame
                cnt = 1

            # name = "{}frame_{:03d}.jpg".format(self.output_path, frame_no)
            # cv2.imwrite(name, return_img)

        else:

            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(self.net_name)
                saver.restore(sess, tf.train.latest_checkpoint('./'))

                input1 = sess.graph.get_tensor_by_name("input1:0")
                input2 = sess.graph.get_tensor_by_name("input2:0")
                output1 = sess.graph.get_tensor_by_name("result_decoded/Tanh:0")

                while True:

                    if self.test_method == "human":

                        if cv2.waitKey(1) == ord('s'):
                            while True:
                                if cv2.waitKey(1) == ord('s'):
                                    break
                        if cv2.waitKey(1) == ord('q'):
                            break

                        ref_img = self.data_provider.get_ref_frame()
                        frame = self.data_provider()

                        return_img = sess.run(output1[0], feed_dict={input1: [ref_img], input2: [frame]})
                        cv2.imshow("output", return_img)

                    if self.test_method == "file":

                        if not self.data_provider.is_first_pass():
                            break

                        frame_no = self.data_provider.get_current_frame_no()

                        ref_img = self.data_provider.get_ref_frame()
                        frame = self.data_provider()

                        return_img = sess.run(output1[0], feed_dict={input1: [ref_img], input2: [frame]}) * self.display_threshold

                        #return_img = cv2.normalize(return_img, return_img, 0, 255, cv2.NORM_MINMAX)
                        name = "{}frame_{:03d}.jpg".format(self.output_path, frame_no)
                        cv2.imwrite(name, return_img)

                print("Tests finished!")
