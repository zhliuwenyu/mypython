import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name = "b")
result = tf.add(a, b, name = "add")
#print(result)
sess = tf.InteractiveSession()
print(result.eval())

w1 = tf.Variable(tf.random_normal([2, 3], mean=2.0, stddev=2.0))
w2 = tf.Variable(tf.random_normal([3, 1], mean=1, stddev=1.0))
x = tf.constant([[0.7, 0.9]])

a1 = tf.matmul(x, w1)
y = tf.matmul(a1, w2)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(y))
sess.close()

print("--------------")

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1))

x = tf.placeholder(tf.float32, shape=(3, 2), name = "input")
a = tf.matmul(x, w1)
b = tf.matmul(a, w2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(b, feed_dict = {x:[[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

