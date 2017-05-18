import tensorflow as tf
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

x = tf.placeholder(tf.float32, shape=[3, 2], name="x-input")
w = tf.Variable(tf.random_normal([2, 1], stddev=1.0, seed=1))
y = tf.matmul(x, w)

weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))