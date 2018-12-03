import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# To plot the figure of accuracy values
acc_values = []

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def digit_recognition():
    # Input
    x_1d = tf.placeholder(tf.float32, [None, 784], name='input')

    # Reshape for convolution recognition:
    # [batch size; height; width; channels]
    x_1d_image = tf.reshape(x_1d, [-1, 28, 28, 1])

    # Convolutional Layer 1:
    # [height; width; input channels; output channels]
    with tf.name_scope("conv_1"):
        w_lr_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        b_lr_1 = tf.Variable(tf.constant(0.1, shape=[32]))

        conv_lr_1 = tf.nn.conv2d(x_1d_image, w_lr_1, strides=[1, 1, 1, 1], padding='SAME') + b_lr_1
        conv_lr_1 = tf.nn.relu(conv_lr_1)

    # Pooling Layer 1
    with tf.name_scope("pool_1"):
        pool_lr_1 = tf.layers.max_pooling2d(conv_lr_1, 2, 2)

    # Convolutional Layer 2
    with tf.name_scope("conv_2"):
        w_lr_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b_lr_2 = tf.Variable(tf.constant(0.1, shape=[64]))

        conv_lr_2 = tf.nn.conv2d(pool_lr_1, w_lr_2, strides=[1, 1, 1, 1], padding='SAME') + b_lr_2
        conv_lr_2 = tf.nn.relu(conv_lr_2)

    # Pooling Layer 2
    with tf.name_scope("pool_2"):
        pool_lr_2 = tf.layers.max_pooling2d(conv_lr_2, 2, 2)

        # Reshaping
        pool_lr_2_rshpd = tf.reshape(pool_lr_2, [-1, 7 * 7 * 64])

    # Fully connected layer
    with tf.name_scope("fully_conn"):
        w_full_conn = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
        b_full_conn = tf.Variable(tf.constant(0.1, shape=[1024]))

        full_conn_lr = tf.matmul(pool_lr_2_rshpd, w_full_conn)
        full_conn_lr = tf.nn.relu(full_conn_lr + b_full_conn)
        full_conn_lr = tf.layers.dropout(full_conn_lr, 0.4)

    # Output, class prediction
    with tf.name_scope('output'):
        w_out = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b_out = tf.Variable(tf.constant(0.1, shape=[10]))

        y = tf.nn.softmax(tf.matmul(full_conn_lr, w_out) + b_out)

    y_ = tf.placeholder(tf.float32, [None, 10], name='ground_truth')

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')
    train_step = tf.train.AdamOptimizer(0.001, name='optimizer').minimize(cross_entropy)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(200):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x_1d: batch_xs,
                                            y_: batch_ys
                                            })
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            indices = np.random.choice(10000, size=1000)
            epoch_acc = sess.run(acc, feed_dict={x_1d: mnist.test.images[indices],
                                                 y_: mnist.test.labels[indices],
                                                 })
            print("Accuracy of %s epoch is %s" % (i, epoch_acc))
            acc_values.append(epoch_acc * 100)

        indices = np.random.choice(10000, size=5000)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        final_acc = sess.run(acc, feed_dict={x_1d: mnist.test.images[indices],
                                             y_: mnist.test.labels[indices],
                                             })
        print("Total accuracy is %s" % final_acc)

    data = np.array(acc_values)
    print(acc_values)
    plt.plot(data)
    plt.show()


digit_recognition()
