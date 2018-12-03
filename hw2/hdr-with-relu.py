import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def with_relu():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784], name='input')

    # 1st hidden relu layer
    w_1 = tf.Variable(tf.truncated_normal(shape=[784, 784], stddev=0.1), name='weight_1')
    b_1 = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1), name='bias_1')

    # dropout, for dropping some neurons because of over fitting
    hidden_layer_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w_1) + b_1), keep_prob=tf.constant(0.5))

    # 2nd layer
    w_2 = tf.Variable(tf.zeros(shape=[784, 10]), name='weight_2')
    b_2 = tf.Variable(tf.zeros(shape=[10]), name='bias_2')

    y = tf.nn.softmax(tf.matmul(hidden_layer_1, w_2) + b_2)

    y_ = tf.placeholder(tf.float32, [None, 10], name='class')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')

    train_step = tf.train.GradientDescentOptimizer(0.5, name='optimizer').minimize(cross_entropy)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                            })

        # output
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels,
                                                             }))

        prediction = (sess.run(y, feed_dict={x: mnist.test.images[1:2]}))
        for index, r in enumerate(prediction):
            print(index, r)
            print('Label is:', mnist.test.labels[1:2])


with_relu()
