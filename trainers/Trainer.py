import tensorflow as tf


class MbTrainer:
    def __init__(self, model, data_provider, output_net_name, logs_dir):
        self.model = model
        self.output_net_name = output_net_name
        self.logs_dir = logs_dir
        self.data_provider = data_provider

    def train(self, num_epochs=20, num_all_steps=1000):

        io_ops = self.model.init_structure()

        # supervised approach
        # cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=output, labels=target))

        # unsupervised approach?
        cost = tf.reduce_mean(self.model.get_value_to_minimize())

        # setup minimizer
        train_step = tf.train.AdamOptimizer(1e-7).minimize(cost)

        saver = tf.train.Saver()
        tf.summary.scalar("cost", cost)

        # Start training
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.logs_dir, graph=sess.graph)

            # Run the initializer
            sess.run(tf.global_variables_initializer())

            # Training cycle
            for step in range(1, num_all_steps):

                data = self.data_provider()
                train_img = data[0]
                # train_out = data[2]

                if step % num_epochs == 0:

                    # return_val = self.model(sess, [train_img[:, 0], train_img[:, 1], train_out], [train_step, cost])
                    return_val = self.model(sess, [train_img[:, 0], train_img[:, 1]], [train_step, cost, io_ops])
                    print("Step: " + str(step) + " Cost: " + str(return_val[0][1]))

                else:
                    # summary = self.model(sess, [train_img[:, 0], train_img[:, 1], train_out], [merged])
                    summary = self.model(sess, [train_img[:, 0], train_img[:, 1]], [merged])
                    writer.add_summary(summary[0][0], step)

            print("Optimization Finished!")

            # Save your model
            saver.save(sess, self.output_net_name)
