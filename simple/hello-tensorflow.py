import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')

sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)

print(sess.run(a + b))

w = tf.Variable(tf.random_normal([3, 2], mean=0.0, stddev=0.4), name='weights')
b = tf.Variable(tf.zeros([2]), name='biases')

with tf.device('/gpu:0'):
    w = tf.Variable(tf.random_normal([3, 2], mean=0.0, stddev=0.4), name='weights')

with tf.device('/job:ps/task:0'):
    w = tf.Variable(tf.random_normal([3, 2], mean=0.0, stddev=0.4), name='weights')

init = tf.global_variables_initializer()

w2 = tf.Variable(w.initialized_value(), name='w2')

# saver = tf.train.Saver({'w2': w2})
# saved = saver.save(sess, 'model.ckpt')
# saver.restore('model.ckpt')
