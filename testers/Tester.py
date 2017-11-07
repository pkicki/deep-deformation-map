import tensorflow as tf
import cv2


class MBTester:
    def __init__(self, data_provider, net_name, output_path="", test_method="human", display_threshold=255):
        self.net_name = net_name
        self.data_provider = data_provider
        self.output_path = output_path
        self.display_threshold = display_threshold

        if test_method in ["human", "file"]:
            self.test_method = test_method
        else:
            raise SyntaxError("No such test method")

    def test(self):

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
