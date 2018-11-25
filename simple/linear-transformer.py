import tensorflow as tf

# todo: smth is wrong


def linear_transform(vec, shape):
    with tf.variable_scope('transform'):
        w = tf.get_variable('matrix', shape, tf.random_normal_initializer())
        return tf.matmul(vec, w)


shape = (2, 2)
vec1 = tf.Variable(tf.random_normal([100], mean=0.0, stddev=0.4), name='vec1')
vec2 = tf.Variable(tf.random_normal([100], mean=0.0, stddev=0.4), name='vec2')
# vec1 = tf.Variable(tf.ones(shape), 'vec1')
# vec2 = tf.Variable(tf.ones(shape), 'vec2')


with tf.variable_scope('linear_transformers') as scope:
    result1 = linear_transform(vec1, shape)
    scope.reuse_variables()
    result2 = linear_transform(vec2, shape)
