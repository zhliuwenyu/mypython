from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/home/liuwenyu/git/pythonws/mypython/MNIST_data", one_hot=True)

print("Training data size:" + str(mnist.train.num_examples))

print "Validatring data size:", mnist.validation.num_examples

print "Testring data size: ", mnist.test.num_examples

print "Example training data label:", mnist.train.labels[0]

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)

print "X shape: ", xs.shape

print "Y shape: ", ys.shape