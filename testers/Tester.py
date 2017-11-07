import tensorflow as tf
import cv2


class MBTester:
    def __init__(self, data_provider, net_name):
        self.net_name = net_name
        self.data_provider = data_provider

    def test(self):

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.net_name)
            saver.restore(sess, tf.train.latest_checkpoint('./'))

            input1 = sess.graph.get_tensor_by_name("input1:0")
            input2 = sess.graph.get_tensor_by_name("input2:0")
            output1 = sess.graph.get_tensor_by_name("result_decoded/Tanh:0")

            while True:

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

